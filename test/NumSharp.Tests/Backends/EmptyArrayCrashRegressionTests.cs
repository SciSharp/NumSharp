using System;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     Regression tests for crashes/corruption discovered while battle-testing np.pad
    ///     against NumPy 2.4.2. Each of these previously segfaulted, divided by zero, or
    ///     threw on arrays with a zero-size dimension. They are reduction/cast/binary-op
    ///     root causes that np.pad's stat and linear_ramp modes surfaced.
    ///
    ///     NumPy reference behaviour (2.4.2) is noted inline.
    /// </summary>
    [TestClass]
    public class EmptyArrayCrashRegressionTests
    {
        // ---------------- np.median / quantile / percentile on empty ----------------

        [TestMethod]
        public void Median_EmptyVector_ReturnsNaN()
        {
            // NumPy: np.median([]) -> nan (previously segfaulted in the quantile IL kernel).
            var r = np.median(np.array(new double[0]));
            double.IsNaN(Convert.ToDouble(r.GetAtIndex(0))).Should().BeTrue();
        }

        [TestMethod]
        public void Median_EmptyAxis_ReturnsNaNFilled()
        {
            // NumPy: np.median(np.zeros((2,0)), axis=1) -> [nan, nan]
            var r = np.median(np.zeros(new Shape(2, 0), typeof(double)), axis: 1);
            r.shape.Should().BeEquivalentTo(new int[] { 2 });
            double.IsNaN(r.GetDouble(0)).Should().BeTrue();
            double.IsNaN(r.GetDouble(1)).Should().BeTrue();
        }

        [TestMethod]
        public void Median_EmptyIntVector_ReturnsNaN()
        {
            // Integer median promotes to float64; empty -> nan (no crash).
            var r = np.median(np.array(new int[0]));
            double.IsNaN(Convert.ToDouble(r.GetAtIndex(0))).Should().BeTrue();
        }

        [TestMethod]
        public void Quantile_EmptyVector_Throws()
        {
            // NumPy diverges from median here: np.quantile([], 0.5) raises IndexError.
            // NumSharp must throw (not segfault).
            Action act = () => np.quantile(np.array(new double[0]), 0.5);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Percentile_EmptyVector_Throws()
        {
            Action act = () => np.percentile(np.array(new double[0]), 50);
            act.Should().Throw<Exception>();
        }

        // ---------------- astype / Cast on a zero-size-dimension array ----------------

        [TestMethod]
        public void Astype_ZeroSizeDimension_PreservesShapeAndRetypes()
        {
            // (1,0) int32 -> double. Previously threw ArgumentOutOfRangeException in CastTo
            // because Shape.IsEmpty only catches the uninitialized sentinel, not a real
            // shape with a zero-size dimension.
            var a = np.zeros(new Shape(1, 0), typeof(int));
            var r = a.astype(typeof(double));
            r.shape.Should().BeEquivalentTo(new int[] { 1, 0 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Astype_ZeroSizeDimension_SlicedView()
        {
            var a = np.zeros(new Shape(4, 3, 0), typeof(int))[new Slice(1, 2), Slice.All, Slice.All];
            var r = a.astype(typeof(double));
            r.shape.Should().BeEquivalentTo(new int[] { 1, 3, 0 });
            r.dtype.Should().Be(typeof(double));
        }

        // ---------------- binary op with a zero-element broadcast result ----------------

        [TestMethod]
        public void BinaryOp_EmptyBroadcast_ReturnsEmptyResult()
        {
            // (3,1,1) * (1,0,2) broadcasts to (3,0,2) — zero elements. The IL/NDIter
            // element-wise path previously corrupted the heap; it must short-circuit.
            var a = np.ones(new Shape(3, 1, 1), typeof(double));
            var b = np.zeros(new Shape(1, 0, 2), typeof(double));
            var r = a * b;
            r.shape.Should().BeEquivalentTo(new int[] { 3, 0, 2 });
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void BinaryOp_EmptyBroadcast_Subtract2D()
        {
            var a = np.ones(new Shape(3, 1), typeof(double));
            var b = np.zeros(new Shape(1, 0), typeof(double));
            var r = a - b;
            r.shape.Should().BeEquivalentTo(new int[] { 3, 0 });
            r.size.Should().Be(0);
        }

        // ---------------- arr[slice] = emptyArray (SetData) ----------------

        [TestMethod]
        public void SetData_AssignEmptyIntoEmptyRegion_NoCrash()
        {
            // np.pad's _PadSimple does `padded[originalSlice] = array` where array.size == 0
            // when padding the non-empty axis of a zero-dim array. Previously divide-by-zero
            // in UnmanagedStorage.SetData (subShape.size % valueshape.size).
            var padded = np.zeros(new Shape(8, 0), typeof(double));
            Action act = () => { padded[new Slice(3, 5), new Slice(0, 0)] = np.ones(new Shape(2, 0), typeof(double)); };
            act.Should().NotThrow();
        }

        // ---------------- basic indexing of an array with a zero-sized axis ----------------
        //
        // A valid integer index into an array whose SIZE is 0 used to throw
        // `IndexOutOfRangeException: The offset 0 is out of range in Shape 0`. The offset
        // bounds-check in Shape.GetSubshape compares against `bufferSize > 0 ? bufferSize : size`,
        // which is 0 for an empty array — so every offset, the correct one included, was "out of
        // range". Per-axis validation (NumPy's real rule) already happened earlier in
        // Shape.InferNegativeCoordinates, so the offset test is only a buffer backstop, and there
        // is no buffer to overrun when the sub-view addresses no element.

        [TestMethod]
        public void Index_TrailingZeroAxis_ReturnsEmptyView()
        {
            // NumPy: a[0] -> (3,0); a[0,0] -> (0,); a[1,2] -> (0,)
            var a = np.zeros(new Shape(2, 3, 0), typeof(double));

            a[0].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[1].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[-1].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[0, 0].shape.Should().BeEquivalentTo(new long[] { 0 });
            a[1, 2].shape.Should().BeEquivalentTo(new long[] { 0 });
            a.GetData(new long[] { 0 }).shape.Should().BeEquivalentTo(new long[] { 3, 0 });

            np.zeros(new Shape(3, 0), typeof(double))[2].shape.Should().BeEquivalentTo(new long[] { 0 });
        }

        [TestMethod]
        public void Index_TrailingZeroAxis_StillRejectsOutOfRange()
        {
            // The backstop was relaxed, not removed — NumPy's per-axis IndexError must survive.
            var a = np.zeros(new Shape(2, 3, 0), typeof(double));

            new Action(() => { var _ = a[2]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[-3]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[0, 3]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[0, 0, 0]; }).Should().Throw<IndexError>("axis 2 has size 0");
            // a zero in the LEADING axis makes even index 0 invalid, as in NumPy
            new Action(() => { var _ = np.zeros(new Shape(0, 3, 4), typeof(double))[0]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_EmptyWindowOverALiveBuffer_ReturnsEmptyView()
        {
            // (5,0) carved out of a live 10-element buffer: the strides still step across the
            // original rows, so [i] computes an offset of 2i into an InternalArray already
            // narrowed to Count 0. NumPy returns the empty view for every valid i.
            var big = np.arange(10).astype(NPTypeCode.Double).reshape(5, 2);
            var window = big[":, 1:1"];
            window.shape.Should().BeEquivalentTo(new long[] { 5, 0 });

            for (int i = 0; i < 5; i++)
                window[i].shape.Should().BeEquivalentTo(new long[] { 0 }, $"window[{i}]");

            new Action(() => { var _ = window[5]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_NonEmptyNeighbours_AreUnaffected()
        {
            // The relaxed backstop must not let a real overrun through on ordinary arrays.
            var big = np.arange(10).astype(NPTypeCode.Double).reshape(5, 2);

            Convert.ToDouble(big[0].GetAtIndex(0)).Should().Be(0d);
            Convert.ToDouble(big[3].GetAtIndex(1)).Should().Be(7d);
            big[4].shape.Should().BeEquivalentTo(new long[] { 2 });
            Convert.ToDouble(big[4, 1].GetAtIndex(0)).Should().Be(9d);

            new Action(() => { var _ = big[5]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = big[0, 2]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_EmptySubView_IsAUsableArray()
        {
            // Not just a shape: the result has to behave like NumPy's empty view.
            var sub = np.zeros(new Shape(2, 3, 0), typeof(double))[0];

            sub.size.Should().Be(0);
            Convert.ToDouble(np.sum(sub).GetAtIndex(0)).Should().Be(0d, "NumPy: sum of empty is 0");
            (sub + sub).shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            sub.T.shape.Should().BeEquivalentTo(new long[] { 0, 3 });
        }

        // ---------------- FANCY (advanced) indexing of an array with a zero-sized axis ----------------
        //
        // The sibling of the basic-indexing block above, in a DIFFERENT guard. Fancy indexing
        // flattens each gathered coordinate to one buffer offset and compares it against
        // LargestReachableOffset(shape, size), which for a contiguous empty source is
        // `size - 1` == -1 — below every offset it is compared against, so every gather into a
        // zero-sized source threw `IndexOutOfRangeException`, the valid ones included.
        //
        // That comparison is NOT what makes a fancy index valid: ScanFancyBounds
        // (NDArray.Indexing.PrepareIndex.cs, NumPy's mapping.c order) and PrepareIndexGetters
        // already check every value against ITS OWN axis extent, -dim <= idx < dim. The offset
        // compare is only the memory-safety backstop for the dereference that follows, and a
        // zero-sized source dereferences nothing: the empty extent is either an INDEXED axis
        // (where the per-axis layer rejects every possible value) or a TRAILING one (making
        // every gathered block subShapeSize == 0). So the UPPER half is disengaged when the
        // source holds no element; the `< 0` half — which catches an offset underflowing below
        // the buffer base — stays live unconditionally.
        //
        // NumPy 2.4.2 reference output is quoted per assertion.

        [TestMethod]
        public void FancyIndex_TrailingZeroAxis_ReturnsEmptyResult()
        {
            // NumPy: np.zeros((3,0))[np.array([0,1])] -> (2,0)
            var a = np.zeros(new Shape(3, 0), typeof(double));

            a[np.array(new long[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 });
            a[np.array(new long[] { -3 })].shape.Should().BeEquivalentTo(new long[] { 1, 0 });
            a[np.array(new long[] { 0, 1, 2, -1, -2, -3 })].shape.Should().BeEquivalentTo(new long[] { 6, 0 });

            // a 2-D index array contributes its own shape: (2,2) + trailing (0,) -> (2,2,0)
            a[np.array(new long[] { 2, -3, 0, 0 }).reshape(2, 2)].shape
                .Should().BeEquivalentTo(new long[] { 2, 2, 0 });

            // 3-D source, 2-D index: (4,1) + trailing (3,0) -> (4,1,3,0)
            var a3 = np.zeros(new Shape(2, 3, 0), typeof(double));
            a3[np.array(new long[] { -1, 1, 0, 0 }).reshape(4, 1)].shape
                .Should().BeEquivalentTo(new long[] { 4, 1, 3, 0 });
            a3[np.array(new long[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 3, 0 });

            // multi-axis fancy stopping short of the empty axis: (1,) + trailing (0,) -> (1,0)
            a3[np.array(new long[] { 0 }), np.array(new long[] { 1 })].shape
                .Should().BeEquivalentTo(new long[] { 1, 0 });
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_ResultIsAUsableArray()
        {
            // NumPy hands back a fresh, C-contiguous, zero-element array (base is None) — not a view.
            var r = np.zeros(new Shape(3, 0), typeof(double))[np.array(new long[] { 0, 1 })];

            r.size.Should().Be(0);
            r.dtype.Should().Be(typeof(double));
            Convert.ToDouble(np.sum(r).GetAtIndex(0)).Should().Be(0d, "NumPy: sum of empty is 0");
            (r + r).shape.Should().BeEquivalentTo(new long[] { 2, 0 });
            r.T.shape.Should().BeEquivalentTo(new long[] { 0, 2 });
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_StillRejectsOutOfRange()
        {
            // The backstop was disengaged, not the validation: NumPy's per-axis IndexError survives,
            // and it is reported against the axis the index acts on — not the flat offset.
            var a = np.zeros(new Shape(3, 0), typeof(double));

            new Action(() => { var _ = a[np.array(new long[] { 5 })]; })
                .Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");
            new Action(() => { var _ = a[np.array(new long[] { 3 })]; })
                .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            new Action(() => { var _ = a[np.array(new long[] { -4 })]; })
                .Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");

            // indexing INTO the zero-sized axis is invalid for every value, 0 included
            new Action(() => { var _ = a[np.array(new long[] { 0 }), np.array(new long[] { 0 })]; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 1 with size 0");

            // a zero in the LEADING (indexed) axis makes even index 0 invalid, as in NumPy
            new Action(() => { var _ = np.zeros(new Shape(0, 3), typeof(double))[np.array(new long[] { 0 })]; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            new Action(() => { var _ = np.zeros(new Shape(0), typeof(double))[np.array(new long[] { 0 })]; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");

            var a3 = np.zeros(new Shape(2, 3, 0), typeof(double));
            new Action(() => { var _ = a3[np.array(new long[] { 0 }), np.array(new long[] { 1 }), np.array(new long[] { 0 })]; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 2 with size 0");
            new Action(() => { var _ = a3[np.array(new long[] { 0 }), np.array(new long[] { 3 })]; })
                .Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 1 with size 3");
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_EmptyIndexAndMaskStillWork()
        {
            // Controls: these two paths never hit the offset guard and must not move.
            var a = np.zeros(new Shape(3, 0), typeof(double));

            // NumPy: an EMPTY index array validates nothing (there are no values to check),
            // so it is legal even where every value would be out of bounds.
            a[np.array(new long[0])].shape.Should().BeEquivalentTo(new long[] { 0, 0 });
            np.zeros(new Shape(0, 3), typeof(double))[np.array(new long[0])].shape
                .Should().BeEquivalentTo(new long[] { 0, 3 });

            // boolean mask path
            a[np.array(new[] { true, false, true })].shape.Should().BeEquivalentTo(new long[] { 2, 0 });
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedStridedView_ReturnsEmptyResult()
        {
            // A (3,0) window carved out of a LIVE 12-element buffer: the offsets are non-zero and
            // step across real rows, so this exercises the guard on a source whose shape is empty
            // but whose buffer is not. NumPy: both -> (2,0).
            var big = np.arange(12).reshape(3, 4);

            big[Slice.All, new Slice(1, 1)][np.array(new long[] { 0, 2 })].shape
                .Should().BeEquivalentTo(new long[] { 2, 0 });
            big["::-1"][Slice.All, new Slice(2, 2)][np.array(new long[] { 0, 2 })].shape
                .Should().BeEquivalentTo(new long[] { 2, 0 }, "negative row stride");

            new Action(() => { var _ = big[Slice.All, new Slice(1, 1)][np.array(new long[] { 5 })]; })
                .Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");

            // the live buffer behind the empty window is untouched
            Convert.ToInt64(big[2, 3].GetAtIndex(0)).Should().Be(11L);
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_AllIndexDtypes()
        {
            // NumPy accepts every integer dtype as a fancy index; none of them may trip the guard.
            var a = np.zeros(new Shape(3, 0), typeof(double));

            a[np.array(new[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "int32");
            a[np.array(new short[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "int16");
            a[np.array(new byte[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "uint8");
            a[np.array(new ushort[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "uint16");
            a[np.array(new uint[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "uint32");
            a[np.array(new ulong[] { 0, 1 })].shape.Should().BeEquivalentTo(new long[] { 2, 0 }, "uint64");
            a[np.array(new sbyte[] { -1 })].shape.Should().BeEquivalentTo(new long[] { 1, 0 }, "int8 negative");
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_SetterAlsoWorks()
        {
            // The setter carries its own copy of the same guard (Selection.Setter.cs), so it had
            // the same defect. NumPy: every one of these is a legal no-op write.
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { 0, 1 })] = (NDArray)5d; })
                .Should().NotThrow();
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { -3 })] = (NDArray)5d; })
                .Should().NotThrow();
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { 2, -3, 0, 0 }).reshape(2, 2)] = (NDArray)5d; })
                .Should().NotThrow();
            new Action(() => { var q = np.zeros(new Shape(2, 3, 0), typeof(double)); q[np.array(new long[] { 0, 1 })] = (NDArray)5d; })
                .Should().NotThrow();
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { 0, 1 })] = np.zeros(new Shape(2, 0), typeof(double)); })
                .Should().NotThrow("value of exactly the indexing-result shape");

            // and the setter's per-axis rejections survive
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { 5 })] = (NDArray)5d; })
                .Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");
            new Action(() => { var q = np.zeros(new Shape(3, 0), typeof(double)); q[np.array(new long[] { 0 }), np.array(new long[] { 0 })] = (NDArray)5d; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 1 with size 0");
            new Action(() => { var q = np.zeros(new Shape(0, 3), typeof(double)); q[np.array(new long[] { 0 })] = (NDArray)5d; })
                .Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
        }

        [TestMethod]
        public void FancyIndex_ZeroSizedSource_SetterReachesTheValueShapeCheck()
        {
            // The premature offset throw MASKED NumPy's real error here: a (2,) value cannot
            // broadcast to the (2,0) indexing result. NumPy raises ValueError with this text;
            // NumSharp used to raise IndexOutOfRangeException about a buffer offset instead.
            var q = np.zeros(new Shape(3, 0), typeof(double));

            new Action(() => { q[np.array(new long[] { 0, 1 })] = np.array(new[] { 1d, 2d }); })
                .Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (2,) could not be broadcast to indexing result of shape (2,0)");
        }

        [TestMethod]
        public void FancyIndex_NonEmptySource_IsUnaffected()
        {
            // The disengagement is keyed off `size == 0`, so ordinary arrays keep the full
            // backstop AND the same values.
            var a = np.arange(12).reshape(3, 4);

            var g = a[np.array(new long[] { 0, 2 })];
            g.shape.Should().BeEquivalentTo(new long[] { 2, 4 });
            Convert.ToInt64(g.GetAtIndex(0)).Should().Be(0L);
            Convert.ToInt64(g.GetAtIndex(7)).Should().Be(11L);

            var m = a[np.array(new long[] { 0, 1 }), np.array(new long[] { 3, 2 })];
            m.shape.Should().BeEquivalentTo(new long[] { 2 });
            Convert.ToInt64(m.GetAtIndex(0)).Should().Be(3L);
            Convert.ToInt64(m.GetAtIndex(1)).Should().Be(6L);

            new Action(() => { var _ = a[np.array(new long[] { 3 })]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[np.array(new long[] { -4 })]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[np.array(new long[] { 0 }), np.array(new long[] { 4 })]; }).Should().Throw<IndexError>();
        }
    }
}
