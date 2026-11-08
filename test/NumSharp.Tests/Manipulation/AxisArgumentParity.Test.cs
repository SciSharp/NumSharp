using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     The axis arguments of the axis-taking functions, against NumPy 2.4.2 (the open-bugs rundown's bug 7 and the
    ///     neighbours its sweep found). Every expected type and text below was probed on NumPy; the oracle tier
    ///     <c>axis_args.jsonl</c> covers the whole matrix (every axis value of 0-d to 3-D operands for every function), and
    ///     these tests pin the contract's distinct rules, each one able to fail on the code it replaced.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy has ONE axis check, <c>normalize_axis_index(axis, ndim, msg_prefix)</c> (C: <c>check_and_adjust_axis_msg</c>),
    ///     and one tuple check on top of it, <c>normalize_axis_tuple</c>; NumSharp ports both once
    ///     (<see cref="DefaultEngine.check_and_adjust_axis(int, int, string)"/>,
    ///     <see cref="DefaultEngine.normalize_axis_tuple(int[], int, object, bool)"/>). The house <see cref="AxisError"/>
    ///     derives from <see cref="ArgumentOutOfRangeException"/>, whose <see cref="Exception.Message"/> appends
    ///     <c>" (Parameter '…')"</c> and an <c>"Actual value was …"</c> line, so an AxisError text is asserted as a prefix.
    ///     </para>
    ///     <para>
    ///     The hang tests carry a timeout: the wrap-then-test loop this replaced (<c>while (axis &lt; 0) axis += ndim</c>)
    ///     never ended for a negative axis of a 0-d input, so a regression must fail, not stall the run.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class AxisArgumentParityTests
    {
        /// <summary>A (2, 3) float64 operand, 1..6.</summary>
        private static NDArray A => np.arange(1.0, 7.0).reshape(2, 3);

        /// <summary>A (2, 3, 4) float64 operand, 1..24.</summary>
        private static NDArray A3 => np.arange(1.0, 25.0).reshape(2, 3, 4);

        /// <summary>A 0-d float64 operand.</summary>
        private static NDArray S => np.array(1.5);

        // ---- the rundown's bug 7: expand_dims / stack raise AxisError -------------------------------------------------

        /// <summary>
        ///     np.expand_dims and np.stack raise NumPy's AxisError (they raised a bare <see cref="ArgumentException"/> with
        ///     the same text), against the RESULT's rank, the axis as given.
        /// </summary>
        [TestMethod]
        public void ExpandDimsAndStack_RaiseAxisError()
        {
            new Action(() => np.expand_dims(A, 5)).Should().ThrowExactly<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 3*");
            new Action(() => np.stack(new[] { A, A }, 3)).Should().ThrowExactly<AxisError>()
                .WithMessage("axis 3 is out of bounds for array of dimension 3*");
            new Action(() => np.stack(new[] { A, A }, -4)).Should().ThrowExactly<AxisError>()
                .WithMessage("axis -4 is out of bounds for array of dimension 3*");
            // A handler written for the old type still catches it.
            new Action(() => np.stack(new[] { A, A }, 7)).Should().Throw<ArgumentException>();
        }

        /// <summary>
        ///     The tuple form of np.expand_dims runs normalize_axis_tuple: every axis is range-checked before the repeat check
        ///     ("repeated axis", NumPy's ValueError), so a tuple holding both reports the out-of-range axis.
        /// </summary>
        [TestMethod]
        public void ExpandDims_Tuple_RangeBeforeRepeat()
        {
            new Action(() => np.expand_dims(A, new[] { 0, 0 })).Should().ThrowExactly<ValueError>()
                .WithMessage("repeated axis");
            new Action(() => np.expand_dims(A, new[] { 0, 0, 9 })).Should().ThrowExactly<AxisError>()
                .WithMessage("axis 9 is out of bounds for array of dimension 5*");
        }

        /// <summary>
        ///     An empty tuple inserts nothing and returns a NEW view object (NumPy's <c>a.reshape(a.shape)</c>), never the
        ///     input itself.
        /// </summary>
        [TestMethod]
        public void ExpandDims_EmptyTuple_IsAFreshView()
        {
            var a = A;
            var r = np.expand_dims(a, new int[0]);
            ReferenceEquals(r, a).Should().BeFalse();
            r.shape.Should().Equal(new long[] { 2, 3 });
            np.shares_memory(r, a).Should().BeTrue();
        }

        // ---- msg_prefix: NumPy names the argument ---------------------------------------------------------------------

        /// <summary>
        ///     NumPy prefixes the AxisError text with the argument's name where the function normalizes more than one axis:
        ///     moveaxis' source / destination, swapaxes' and diagonal's axis1 / axis2, cross' axisa, unstack's source
        ///     (it is moveaxis), average's and trim_zeros' axis.
        /// </summary>
        [TestMethod]
        public void AxisError_CarriesNumPysArgumentPrefix()
        {
            Expect<AxisError>(() => np.moveaxis(A, 5, 0), "source: axis 5 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.moveaxis(A, 0, -3), "destination: axis -3 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.swapaxes(A, 0, 2), "axis2: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.swapaxes(A, -3, 0), "axis1: axis -3 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.unstack(A, 2), "source: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.diagonal(A, 0, 0, 2), "axis2: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.cross(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 3)), axisa: 2),
                "axisa: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.average(A, 2), "axis: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.trim_zeros(A, "fb", 2), "axis: axis 2 is out of bounds for array of dimension 2");
            // The functions with one axis print no prefix.
            Expect<AxisError>(() => np.rollaxis(A, 5), "axis 5 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.size(A, 2), "axis 2 is out of bounds for array of dimension 2");
        }

        /// <summary>
        ///     The repeat texts of normalize_axis_tuple name the argument too ("repeated axis in `source` argument"); the
        ///     plain text is NumPy's where no argname is passed (median, vector_norm), and the ufunc reductions keep their own
        ///     ("duplicate value in 'axis'").
        /// </summary>
        [TestMethod]
        public void RepeatedAxis_CarriesNumPysTexts()
        {
            Expect<ValueError>(() => np.moveaxis(A3, new[] { 0, 0 }, new[] { 0, 1 }), "repeated axis in `source` argument");
            Expect<ValueError>(() => np.moveaxis(A3, new[] { 0, 1 }, new[] { 1, 1 }), "repeated axis in `destination` argument");
            Expect<ValueError>(() => np.average(A3, new[] { 0, 0 }), "repeated axis in `axis` argument");
            Expect<ValueError>(() => np.median(A3, new[] { 0, -3 }), "repeated axis");
            Expect<ValueError>(() => np.linalg.vector_norm(A3, new[] { 0, 0 }), "repeated axis");
            Expect<ValueError>(() => np.all(A3, new[] { 0, 0 }), "duplicate value in 'axis'");
            Expect<ValueError>(() => np.transpose(A3, new[] { 0, 0, 1 }), "repeated axis in transpose");
            Expect<ValueError>(() => np.transpose(A3, new[] { 0, 1 }), "axes don't match array");
            // Range before repeat, in every one of them.
            Expect<AxisError>(() => np.moveaxis(A3, new[] { 0, 0, 3 }, new[] { 0, 1, 2 }),
                "source: axis 3 is out of bounds for array of dimension 3");
            Expect<AxisError>(() => np.all(A3, new[] { 0, 0, 3 }), "axis 3 is out of bounds for array of dimension 3");
        }

        /// <summary>
        ///     np.moveaxis: a repeated axis is NumPy's repeat ValueError (the old helper REMOVED the repeat and then reported
        ///     a length mismatch), and the real length mismatch has NumPy's text without the stray trailing quote.
        /// </summary>
        [TestMethod]
        public void Moveaxis_RepeatAndLengthTexts()
        {
            Expect<ValueError>(() => np.moveaxis(A3, new[] { 0, 1 }, new[] { 1 }),
                "`source` and `destination` arguments must have the same number of elements");
        }

        /// <summary>normalize_axis_tuple builds a new array: the caller's axes are never rewritten in place (they were).</summary>
        [TestMethod]
        public void NormalizeAxisTuple_DoesNotRewriteTheCallersArray()
        {
            var axes = new[] { -1, 0 };
            var normalized = DefaultEngine.normalize_axis_tuple(axes, 3);
            normalized.Should().Equal(2, 0);
            axes.Should().Equal(-1, 0);
            var source = new[] { -1 };
            np.moveaxis(A3, source, new[] { 0 });
            source.Should().Equal(-1);
        }

        /// <summary>np.rollaxis' start has its own message-only AxisError, reporting start after its negative wrap.</summary>
        [TestMethod]
        public void Rollaxis_StartText()
        {
            Expect<AxisError>(() => np.rollaxis(A, 0, 4), "'start' arg requires -2 <= start < 3, but 4 was passed in");
            Expect<AxisError>(() => np.rollaxis(A, 0, -3), "'start' arg requires -2 <= start < 3, but -1 was passed in");
        }

        /// <summary>np.rollaxis returns a NEW view even when no axis moves (NumPy's <c>a[...]</c>), never the input.</summary>
        [TestMethod]
        public void Rollaxis_Identity_IsAFreshView()
        {
            var a = A;
            var r = np.rollaxis(a, 0, 1);
            ReferenceEquals(r, a).Should().BeFalse();
            np.shares_memory(r, a).Should().BeTrue();
        }

        // ---- silent results and hangs where NumPy raises ----------------------------------------------------------------

        /// <summary>
        ///     An axis of -ndim - 1 is out of range. The wrap-then-test loops accepted it (wrapping twice to the last axis)
        ///     and returned a value where NumPy raises.
        /// </summary>
        [TestMethod]
        public void MinusNdimMinusOne_IsRefused_NotWrappedTwice()
        {
            const string text = "axis -3 is out of bounds for array of dimension 2";
            Expect<AxisError>(() => np.argmax(A, -3), text);
            Expect<AxisError>(() => np.argmin(A, -3), text);
            Expect<AxisError>(() => np.cumprod(A, -3), text);
            Expect<AxisError>(() => np.nancumprod(A, -3), text);
            Expect<AxisError>(() => np.nansum(A, -3), text);
            Expect<AxisError>(() => np.nanprod(A, -3), text);
            Expect<AxisError>(() => np.count_nonzero(A, -3), text);
            Expect<AxisError>(() => np.squeeze(np.zeros(new Shape(1, 3, 1)), -4),
                "axis -4 is out of bounds for array of dimension 3");
        }

        /// <summary>
        ///     A negative axis of a 0-d input: the wrap-then-test loop (<c>axis += ndim</c> with ndim 0) never terminated.
        ///     NumPy refuses it (count_nonzero / squeeze / the reductions against rank 0; argmax / cumprod against the rank
        ///     1 of the raveled input).
        /// </summary>
        [TestMethod]
        [Timeout(20_000)]
        public void ZeroD_NegativeAxis_TerminatesWithNumPysError()
        {
            Expect<AxisError>(() => np.count_nonzero(S, -2), "axis -2 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.squeeze(S, -2), "axis -2 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.max(S, -2), "axis -2 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.argmax(S, -2), "axis -2 is out of bounds for array of dimension 1");
            Expect<AxisError>(() => np.cumprod(S, -2), "axis -2 is out of bounds for array of dimension 1");
        }

        /// <summary>
        ///     The 0-d special case: axis 0 and -1 slip through for the ufunc reductions, count_nonzero and squeeze (no axis
        ///     is selected), and for the PyArray_CheckAxis functions (the input ravels to 1-D first) — with NumPy's values.
        /// </summary>
        [TestMethod]
        public void ZeroD_AxisZero_IsNumPysBackwardCompatibleCase()
        {
            np.count_nonzero(S, 0).GetInt64().Should().Be(1);
            np.squeeze(S, -1).ndim.Should().Be(0);
            np.argmax(S, 0).GetInt64().Should().Be(0);
            np.prod(S, -1).GetDouble().Should().Be(1.5);
        }

        /// <summary>
        ///     A one-element or 0-d reduction answered from its shortcut without reading the axis; NumPy checks the axis
        ///     first (PyUFunc_GenericReduction), so the axis error wins.
        /// </summary>
        [TestMethod]
        public void ZeroD_ReductionShortcuts_CheckTheAxisFirst()
        {
            Expect<AxisError>(() => np.prod(S, 1), "axis 1 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.amax(S, 1), "axis 1 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.amin(S, -2), "axis -2 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.nanmax(S, 1), "axis 1 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.prod(np.array(new[] { 2.0 }), 1), "axis 1 is out of bounds for array of dimension 1");
        }

        /// <summary>
        ///     The PyArray_CheckAxis functions treat a 0-d input as 1-D, so they report "dimension 1" (repeat / packbits said
        ///     0, compress accepted ANY axis of a 0-d input).
        /// </summary>
        [TestMethod]
        public void ZeroD_CheckAxisFunctions_ReportRankOne()
        {
            const string text = "axis 1 is out of bounds for array of dimension 1";
            Expect<AxisError>(() => np.repeat(S, 2, 1), text);
            Expect<AxisError>(() => np.packbits(np.array((byte)1), 1), text);
            Expect<AxisError>(() => np.take(S, np.array(new long[] { 0 }), 1), text);
            Expect<AxisError>(() => np.compress(np.array(new[] { true }), S, 1), text);
            Expect<AxisError>(() => np.argsort(S, 1), text);
            // np.sort does not ravel: a 0-d input has no axis at all.
            Expect<AxisError>(() => np.sort(S, -1), "axis -1 is out of bounds for array of dimension 0");
        }

        // ---- the IndexError NumPy leaks ---------------------------------------------------------------------------------

        /// <summary>
        ///     np.split / np.array_split / np.linalg.cross read <c>a.shape[axis]</c> before anything else, a Python tuple
        ///     index: an axis out of range is IndexError "tuple index out of range" (NumSharp raised an
        ///     ArgumentOutOfRangeException / AxisError), checked before the sections.
        /// </summary>
        [TestMethod]
        public void ShapeIndexFunctions_RaiseNumPysIndexError()
        {
            Expect<IndexError>(() => np.split(A, 1, 2), "tuple index out of range");
            Expect<IndexError>(() => np.split(S, 1), "tuple index out of range");
            Expect<IndexError>(() => np.array_split(A, 0, 5), "tuple index out of range");
            Expect<IndexError>(() => np.linalg.cross(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 3)), 2),
                "tuple index out of range");
            Expect<IndexError>(() => np.fft.irfftn(A, axes: new int[0]), "tuple index out of range");
        }

        // ---- NumPy's ValueError / TypeError types in the same checks ------------------------------------------------------

        /// <summary>
        ///     The neighbouring argument checks raise NumPy's ValueError: a non-unit squeeze axis, two equal diagonal axes,
        ///     rot90's axes, split's sections, the stacking family's empty operand list.
        /// </summary>
        [TestMethod]
        public void NeighbouringChecks_RaiseValueError()
        {
            Expect<ValueError>(() => np.squeeze(np.zeros(new Shape(1, 3, 1)), 1),
                "cannot select an axis to squeeze out which has size not equal to one");
            Expect<ValueError>(() => np.diagonal(A, 0, 0, -2), "axis1 and axis2 cannot be the same");
            Expect<ValueError>(() => np.rot90(A, 1, new[] { 0, 0 }), "Axes must be different.");
            Expect<ValueError>(() => np.split(np.arange(9), 4), "array split does not result in an equal division");
            Expect<ValueError>(() => np.stack(new NDArray[0]), "need at least one array to stack");
            Expect<ValueError>(() => np.vstack(), "need at least one array to concatenate");
            Expect<ValueError>(() => np.hstack(), "need at least one array to concatenate");
            Expect<ValueError>(() => np.dstack(), "need at least one array to concatenate");
            Expect<ValueError>(() => np.concatenate(new NDArray[0]), "need at least one array to concatenate");
            Expect<ValueError>(() => np.diff(S), "diff requires input that is at least one dimensional");
        }

        /// <summary>np.atleast_1d / 2d / 3d of no arrays is NumPy's empty tuple, not an error.</summary>
        [TestMethod]
        public void AtleastNd_OfNoArrays_IsEmpty()
        {
            np.atleast_1d(new NDArray[0]).Should().BeEmpty();
            np.atleast_2d(new NDArray[0]).Should().BeEmpty();
            np.atleast_3d(new NDArray[0]).Should().BeEmpty();
        }

        /// <summary>NumPy's boolean subtract / negative refusals are TypeError (they were NotSupportedException).</summary>
        [TestMethod]
        public void BooleanSubtractAndNegative_RaiseTypeError()
        {
            var b = np.array(new[] { true, false });
            Expect<TypeError>(() => np.subtract(b, b),
                "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
            Expect<TypeError>(() => np.negative(b),
                "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.");
        }

        // ---- values the axis rules decide -------------------------------------------------------------------------------

        /// <summary>
        ///     np.linalg.vector_norm over a TUPLE of axes merges them into one and takes the VECTOR norm. A 2-tuple used to
        ///     reach norm's MATRIX branch (the spectral norm for the default order), a 3-tuple raised.
        /// </summary>
        [TestMethod]
        public void VectorNorm_TupleAxis_IsTheVectorNormOfTheMergedAxes()
        {
            var r = np.linalg.vector_norm(A3, new[] { 0, 1 });
            r.shape.Should().Equal(new long[] { 4 });
            r.ToArray<double>().Should().Equal(31.71750305430741, 33.823069050575526, 35.9722114972099, 38.157568056677825);
            np.linalg.vector_norm(A3, new[] { 0, 1, 2 }).GetDouble().Should().Be(70.0);
            np.linalg.vector_norm(A3, new[] { 0, 1 }, ord: 1).ToArray<double>().Should().Equal(66.0, 72.0, 78.0, 84.0);
            var kd = np.linalg.vector_norm(A3, new[] { 2, 0 }, keepdims: true);
            kd.shape.Should().Equal(new long[] { 1, 3, 1 });
            kd.ToArray<double>().Should().Equal(29.597297173897484, 39.344631145812, 49.759421218498915);
            var none = np.linalg.vector_norm(A, new int[0]);
            none.shape.Should().Equal(new long[] { 2, 3 });
            none.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        }

        /// <summary>
        ///     The quantile family: an EMPTY axis tuple reduces nothing (each element is its own slice; it reduced every
        ///     axis), a complex input is NumPy's TypeError before anything else (it computed a value), and the axis is
        ///     checked before a bool input meets the subtract.
        /// </summary>
        [TestMethod]
        public void QuantileFamily_AxisAndDtypeRules()
        {
            var med = np.median(A, new int[0]);
            med.shape.Should().Equal(new long[] { 2, 3 });
            med.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            var q = np.quantile(A, 0.5, new int[0]);
            q.shape.Should().Equal(new long[] { 2, 3 });
            Expect<TypeError>(() => np.quantile(np.array(new[] { new System.Numerics.Complex(1, 1), new System.Numerics.Complex(2, 0) }), 0.5),
                "a must be an array of real numbers");
            Expect<TypeError>(() => np.nanpercentile(np.array(new[] { new System.Numerics.Complex(1, 1) }), 50.0),
                "a must be an array of real numbers");
            Expect<AxisError>(() => np.quantile(np.array(new[] { true, false }), 0.5, 5),
                "axis 5 is out of bounds for array of dimension 1");
        }

        /// <summary>np.gradient with a tuple axis and no spacing (a null spacing) is NumPy's <c>np.gradient(f, axis=(…))</c>.</summary>
        [TestMethod]
        public void Gradient_TupleAxis_NullSpacing()
        {
            NDArray[] g = np.gradient(A3, (object)null, new[] { 0, 2 });
            g.Length.Should().Be(2);
            g[1].shape.Should().Equal(new long[] { 2, 3, 4 });
            g[1].ToArray<double>().All(v => v == 1.0).Should().BeTrue();
        }

        /// <summary>
        ///     Shape.GetAxis (the dimensions a reduction over one axis leaves) normalizes a negative axis as NumPy does: -2
        ///     is the second-to-last axis. It mapped EVERY negative axis to the last one.
        /// </summary>
        [TestMethod]
        public void ShapeGetAxis_NormalizesANegativeAxis()
        {
            Shape.GetAxis(new long[] { 2, 3, 4 }, -2).Should().Equal(2L, 4L);
            Shape.GetAxis(new long[] { 2, 3, 4 }, -3).Should().Equal(3L, 4L);
            new Action(() => Shape.GetAxis(new long[] { 2, 3, 4 }, -4)).Should().ThrowExactly<AxisError>()
                .WithMessage("axis -4 is out of bounds for array of dimension 3*");
        }

        /// <summary>
        ///     np.pad's dict pad_width (NumPy 2.3+) is NumPy's dict branch: the keys are LIST INDICES (an axis outside
        ///     <c>[-ndim, ndim)</c> is the list's IndexError, never an AxisError; a 0-d array's list has no slot), each width
        ///     is matched BEFORE its axis is assigned (anything but an int or a 2-tuple of ints is typing.assert_never's
        ///     AssertionError, a typed array included), the table's dtype comes next (TypeError) and a negative width last,
        ///     after every key (ValueError). The old path raised the base ArgumentException for all of them, checked the axis
        ///     first, and let a negative side of a tuple through to the slicing.
        /// </summary>
        [TestMethod]
        public void Pad_DictPadWidth_FollowsNumPysStatements()
        {
            Expect<IndexError>(() => np.pad(A, Widths((5, 1))), "list assignment index out of range");
            Expect<IndexError>(() => np.pad(A, Widths((-3, 1))), "list assignment index out of range");
            Expect<AssertionError>(() => np.pad(A, Widths((5, "x"))), "Expected code to be unreachable, but got: 'x'");
            Expect<AssertionError>(() => np.pad(A, Widths((0, 1.5))), "Expected code to be unreachable, but got: 1.5");
            Expect<AssertionError>(() => np.pad(A, Widths((0, new long[] { 1, 2 }))),
                "Expected code to be unreachable, but got: array([1, 2])");
            Expect<IndexError>(() => np.pad(A, Widths((0, -1), (5, 1))), "list assignment index out of range");
            Expect<ValueError>(() => np.pad(A, Widths((0, (1, -2)))), "index can't contain negative values");
            Expect<TypeError>(() => np.pad(np.arange(3.0), Widths((0, true))), "`pad_width` must be of integral type.");
            Expect<TypeError>(() => np.pad(A, Widths((0, System.Numerics.BigInteger.Pow(2, 63)))),
                "`pad_width` must be of integral type.");
            Expect<TypeError>(() => np.pad(S, Widths()), "`pad_width` must be of integral type.");
            Expect<IndexError>(() => np.pad(S, Widths((0, 1))), "list assignment index out of range");
            // A bool is a Python int: True pads by one while the untouched row keeps the table an integer one.
            np.pad(A, Widths((0, true))).shape.Should().Equal(4L, 3L);
            // A later key for the same axis wins, as a later list assignment does.
            np.pad(A, Widths((0, 1), (-2, 2))).shape.Should().Equal(6L, 3L);
        }

        /// <summary>
        ///     np.evaluate's reductions follow the axis rule of the NumPy function each one stands for. The ufunc reductions
        ///     (and the count_nonzero / nansum / nanprod / nanmean twins, which read the axis through np.sum) take axis 0 / -1
        ///     of a 0-d child as no axis, the flat answer; they raised AxisError there. np.mean / np.var / np.std count the
        ///     reduced items with normalize_axis_index and refuse it, and np.average names its argument in the AxisError
        ///     (np.evaluate's average omitted the prefix).
        /// </summary>
        [TestMethod]
        public void Evaluate_Reductions_FollowTheirNumPyFunctionsAxisRule()
        {
            var s = (NDExpr)np.array(2.5) * 1.0;
            ((double)np.evaluate(NDExpr.Sum(s, 0))).Should().Be(2.5);
            ((double)np.evaluate(NDExpr.Max(s, -1))).Should().Be(2.5);
            ((bool)np.evaluate(NDExpr.Any(s, 0))).Should().BeTrue();
            ((long)np.evaluate(NDExpr.CountNonzero(s, -1))).Should().Be(1L);
            ((double)np.evaluate(NDExpr.NanMean(s, 0))).Should().Be(2.5);
            np.evaluate(NDExpr.Prod(s, 0, keepdims: true)).ndim.Should().Be(0);
            Expect<AxisError>(() => np.evaluate(NDExpr.Sum(s, 1)), "axis 1 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.evaluate(NDExpr.Mean(s, 0)), "axis 0 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.evaluate(NDExpr.Var(s, axis: -1)), "axis -1 is out of bounds for array of dimension 0");
            Expect<AxisError>(() => np.evaluate(NDExpr.Average((NDExpr)A, (NDExpr)A, 2)),
                "axis: axis 2 is out of bounds for array of dimension 2");
            Expect<AxisError>(() => np.evaluate(NDExpr.Average(s, s, 0)),
                "axis: axis 0 is out of bounds for array of dimension 0");
            // A strided operand passes the streamed route's axis check; an EMPTY one skips the stream, so the materialized
            // route checks the axis itself: the same prefix either way.
            var strided = np.arange(1.0, 13.0).reshape(2, 6)[":, ::2"];
            Expect<AxisError>(() => np.evaluate(NDExpr.Average((NDExpr)strided, (NDExpr)strided, 5)),
                "axis: axis 5 is out of bounds for array of dimension 2");
            var empty = np.zeros(new Shape(0, 3));
            Expect<AxisError>(() => np.evaluate(NDExpr.Average((NDExpr)empty * 1.0, (NDExpr)empty * 1.0, -3)),
                "axis: axis -3 is out of bounds for array of dimension 2");
            // The child's rank is its BROADCAST rank: a 0-d leaf beside a 2-D one leaves axis 0 a real axis.
            np.evaluate(NDExpr.Sum((NDExpr)np.array(2.0) * (NDExpr)A, 0)).shape.Should().Equal(3L);
        }

        /// <summary>
        ///     np.linalg.tensorsolve's axes are list VALUES (<c>allaxes.remove(k)</c>), never normalized: a negative axis or
        ///     one past the rank is Python's ValueError "list.remove(x): x not in list" (C#'s List.Remove only returned
        ///     false, and the stray axis surfaced as transpose's "axes don't match array"), and a repeat is legal (the
        ///     re-appended axis is removed again).
        /// </summary>
        [TestMethod]
        public void TensorSolve_AxesAreListValues()
        {
            var a = np.eye(6).reshape(2, 3, 2, 3);
            var b = np.arange(1.0, 7.0).reshape(2, 3);
            Expect<ValueError>(() => np.linalg.tensorsolve(a, b, new[] { -1 }), "list.remove(x): x not in list");
            Expect<ValueError>(() => np.linalg.tensorsolve(a, b, new[] { 4 }), "list.remove(x): x not in list");
            np.linalg.tensorsolve(a, b, new[] { 0, 0 }).shape.Should().Equal(3L, 2L);
        }

        /// <summary>numpy.ma's compress_nd checks its axis first, before returning the data of an unmasked operand.</summary>
        [TestMethod]
        public void MaCompressNd_ChecksTheAxisFirst()
        {
            Expect<AxisError>(() => np.ma.compress_nd(np.ma.array(A), 2), "axis 2 is out of bounds for array of dimension 2");
        }

        /// <summary>
        ///     A dict <c>pad_width</c> in the given order (a <see cref="System.Collections.Generic.Dictionary{TKey,TValue}"/>
        ///     enumerates in insertion order, as Python's dict does, so the order NumPy reads the items in is kept).
        /// </summary>
        /// <param name="items">(axis, width) pairs; a width is boxed as given (an int, a bool, a tuple, a typed array …).</param>
        /// <returns>The dictionary.</returns>
        private static System.Collections.Generic.Dictionary<int, object> Widths(params (int axis, object width)[] items)
        {
            var d = new System.Collections.Generic.Dictionary<int, object>();
            foreach (var (axis, width) in items)
                d[axis] = width;
            return d;
        }

        /// <summary>
        ///     Asserts that <paramref name="call"/> raises exactly <typeparamref name="T"/> with a message that starts with
        ///     <paramref name="text"/> (an AxisError appends .NET's parameter suffix; every other type carries NumPy's text
        ///     whole).
        /// </summary>
        /// <typeparam name="T">The exact exception type NumPy's corresponds to.</typeparam>
        /// <param name="call">The call that must raise.</param>
        /// <param name="text">NumPy's message.</param>
        private static void Expect<T>(Func<object> call, string text) where T : Exception
        {
            var ex = new Action(() => call()).Should().ThrowExactly<T>().Which;
            ex.Message.Should().StartWith(text);
        }

        /// <summary>
        ///     Asserts that <paramref name="call"/> raises exactly <typeparamref name="T"/> with a message that starts with
        ///     <paramref name="text"/> — the overload for calls that return nothing.
        /// </summary>
        /// <typeparam name="T">The exact exception type.</typeparam>
        /// <param name="call">The call that must raise.</param>
        /// <param name="text">NumPy's message.</param>
        private static void Expect<T>(Action call, string text) where T : Exception
        {
            var ex = new Action(call).Should().ThrowExactly<T>().Which;
            ex.Message.Should().StartWith(text);
        }
    }
}
