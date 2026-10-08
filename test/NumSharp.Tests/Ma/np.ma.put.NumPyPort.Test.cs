using System;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's <c>put</c> (<c>MaskedArray.put</c> and the <c>ma.put</c> function) as NumPy's own statements.
    ///     Under a HARD mask NumPy runs <c>mask = self._mask[indices]; indices = narray(indices); values = narray(values,
    ///     subok=True); values.resize(indices.shape); indices = indices[~mask]; values = values[~mask]</c> before
    ///     <c>self._data.put</c>. Every expected value and message is NumPy 2.4.2's output for the same call.
    /// </summary>
    /// <remarks>
    ///     Before the port (2026-10-08) NumSharp applied a hard mask to the flat positions it wrote and restored the masked
    ///     ones afterwards: an N-D target was written where NumPy raises IndexError, short values cycled where NumPy
    ///     zero-pads them, masked values were written where NumPy raises ValueError, and the soft branch left the masked
    ///     constant's slots' data as it was where NumPy writes the constant's 0.0. The <c>ma_put</c> section of the
    ///     <c>ma_select</c> oracle tier covers the ground these tests pin by example.
    /// </remarks>
    [TestClass]
    public class np_ma_put_numpy_port_tests
    {
        private static NDMaskedArray Hard1D() =>
            np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { false, true, false, false }, hard_mask: true);

        private static NDMaskedArray Hard2D() =>
            np.ma.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, mask: new[,] { { false, true, false }, { false, false, true } },
                hard_mask: true);

        private static long[] Data(NDMaskedArray x) => x.data.flatten().ToArray<long>();

        private static bool[] Mask(NDMaskedArray x) => np.ma.getmaskarray(x).flatten().ToArray<bool>();

        private static bool IsNomask(NDMaskedArray x) => ReferenceEquals(x.mask, np.ma.nomask);

        /// <summary>
        ///     The rundown's bug: <c>self._mask[indices]</c> indexes the FIRST axis of an N-D mask, so the boolean array
        ///     <c>~mask</c> has more dimensions than the index array and <c>indices[~mask]</c> raises — before anything is
        ///     written. Each NumPy text below was probed; the array is unchanged after every call.
        /// </summary>
        [TestMethod]
        public void HardMask_NDTarget_RaisesNumPysIndexError_AndWritesNothing()
        {
            void Raises(Action<NDMaskedArray> put, string message)
            {
                var x = Hard2D();
                put.Invoking(p => p(x)).Should().Throw<IndexError>().WithMessage(message);
                Data(x).Should().Equal(1, 2, 3, 4, 5, 6);
                Mask(x).Should().Equal(false, true, false, false, false, true);
            }

            // Rows 0 and 1 exist: the mask lookup works, the boolean index has 2 dimensions.
            Raises(x => x.put(np.array(new[] { 0, 1 }), new long[] { 7, 8 }),
                "too many indices for array: array is 1-dimensional, but 2 were indexed");
            // Index 4 is a flat position, but the lookup indexes axis 0 (2 rows).
            Raises(x => x.put(np.array(new[] { 0, 4 }), new long[] { 7, 8 }), "index 4 is out of bounds for axis 0 with size 2");
            // The lookup ignores the mode.
            Raises(x => np.ma.put(x, np.array(new[] { 0, 9 }), new long[] { 7, 8 }, "wrap"),
                "index 9 is out of bounds for axis 0 with size 2");
            Raises(x => x.put(np.array(new[] { 0, 1 }), new long[] { 7, 8 }, "clip"),
                "too many indices for array: array is 1-dimensional, but 2 were indexed");
            // A scalar index: mask[0] is a row, the index array is 0-d.
            Raises(x => x.put(NDArray.Scalar(0L), 7), "too many indices for array: array is 0-dimensional, but 1 were indexed");
            // A 2-D index array: mask[[[0], [1]]] has 3 dimensions.
            Raises(x => x.put(np.array(new long[,] { { 0 }, { 1 } }), new long[] { 7, 8 }),
                "too many indices for array: array is 2-dimensional, but 3 were indexed");

            // A 3-D target, and the degenerate (1, 4) / (4, 1) shapes: still N-D.
            var x3 = np.ma.array(np.arange(8).reshape(2, 2, 2), mask: np.equal(np.mod(np.arange(8).reshape(2, 2, 2), 3), 0),
                hard_mask: true);
            x3.Invoking(v => v.put(np.array(new[] { 0, 1 }), new long[] { 7, 8 })).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 1-dimensional, but 3 were indexed");
            var row = np.ma.array(new long[,] { { 1, 2, 3, 4 } }, mask: new[,] { { false, true, false, false } }, hard_mask: true);
            row.Invoking(v => v.put(np.array(new[] { 2 }), new long[] { 7 })).Should().Throw<IndexError>()
                .WithMessage("index 2 is out of bounds for axis 0 with size 1");
            var col = np.ma.array(new long[,] { { 1 }, { 2 }, { 3 }, { 4 } }, mask: new[,] { { false }, { true }, { false }, { false } },
                hard_mask: true);
            col.Invoking(v => v.put(np.array(new[] { 0, 2 }), new long[] { 7, 8 })).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 1-dimensional, but 2 were indexed");
            Data(col).Should().Equal(1, 2, 3, 4);

            // An all-False mask ARRAY is still a hard mask (NumPy tests `_mask is not nomask`), so it raises too.
            var allFalse = np.ma.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, mask: new bool[2, 3], hard_mask: true);
            allFalse.Invoking(v => v.put(np.array(new[] { 0, 1 }), new long[] { 7, 8 })).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 1-dimensional, but 2 were indexed");
        }

        /// <summary>
        ///     The values are resized to the indices' shape with <c>ndarray.resize</c>: ZERO-PADDED or truncated, never
        ///     cycled (the soft branch cycles). A scalar becomes a 1-element array, so only the first index gets it.
        /// </summary>
        [TestMethod]
        public void HardMask_1D_ValuesAreZeroPadded_NotCycled()
        {
            var x = Hard1D();
            x.put(np.array(new[] { 0, 1, 2, 3 }), new long[] { 10, 20 });
            Data(x).Should().Equal(10, 2, 0, 0);     // index 1 is masked: dropped with its value 20
            Mask(x).Should().Equal(false, true, false, false);

            x = Hard1D();
            x.put(np.array(new[] { 0, 2, 3 }), 5);
            Data(x).Should().Equal(5, 2, 0, 0);

            x = Hard1D();
            x.put(np.array(new[] { 0, 2 }), new long[] { 10, 20, 30, 40 });
            Data(x).Should().Equal(10, 2, 20, 4);    // truncated

            // An empty values list grows to zeros.
            x = Hard1D();
            x.put(np.array(new[] { 1, 2 }), new double[0]);
            Data(x).Should().Equal(1, 2, 0, 4);

            // The soft branch cycles, as ndarray.put does.
            var soft = np.ma.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, mask: new[,] { { false, true, false }, { false, false, true } });
            soft.put(np.array(new[] { 0, 2, 4 }), new long[] { 7, 8 });
            Data(soft).Should().Equal(7, 2, 8, 4, 7, 6);
        }

        /// <summary>Indices on masked slots are dropped with their values; scalar, negative, 2-D and bool indices.</summary>
        [TestMethod]
        public void HardMask_1D_IndexForms()
        {
            var x = Hard1D();
            x.put(NDArray.Scalar(0L), 5);
            Data(x).Should().Equal(5, 2, 3, 4);
            x = Hard1D();
            x.put(NDArray.Scalar(1L), 5);            // masked: nothing written
            Data(x).Should().Equal(1, 2, 3, 4);

            x = Hard1D();
            x.put(np.array(new[] { -1, -3 }), new long[] { 7, 8 });
            Data(x).Should().Equal(1, 2, 3, 7);      // -3 is index 1, masked

            // A 2-D index array on a 1-D target is fine: the values take its shape.
            x = Hard1D();
            x.put(np.array(new long[,] { { 0, 2 }, { 3, 1 } }), new long[] { 7, 8 });
            Data(x).Should().Equal(7, 2, 8, 0);

            // Bool indices: mask[bool] selects mask entries, then put casts True to 1.
            var h = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { true, false, false, false }, hard_mask: true);
            h.put(np.array(new[] { true, true, true, true }), new long[] { 7, 8, 9, 10 });
            Data(h).Should().Equal(1, 10, 3, 4);
            Hard1D().Invoking(v => v.put(np.array(new[] { true, false, true, false }), new long[] { 7, 8 }))
                .Should().Throw<IndexError>()
                .WithMessage("boolean index did not match indexed array along axis 0; size of axis is 4 but size of corresponding boolean axis is 2");
        }

        /// <summary>
        ///     The mask lookup ignores the mode (an out-of-bounds index raises in wrap and clip too), but the write does
        ///     not: under clip a negative index passes the lookup as the LAST element and is written — and unmasked — at
        ///     index 0. NumPy 2.4.2: <c>[--, 2, 3, 4]</c> hard, <c>put([-1], [7], mode='clip')</c> gives <c>[7, 2, 3, 4]</c>,
        ///     nomask.
        /// </summary>
        [TestMethod]
        public void HardMask_Modes()
        {
            foreach (var mode in new[] { "raise", "wrap", "clip" })
                Hard1D().Invoking(v => v.put(np.array(new[] { 0, 9 }), new long[] { 7, 8 }, mode))
                    .Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 0 with size 4");

            var h = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { true, false, false, false }, hard_mask: true);
            h.put(np.array(new[] { -1 }), new long[] { 7 }, "clip");
            Data(h).Should().Equal(7, 2, 3, 4);
            IsNomask(h).Should().BeTrue();

            h = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { true, false, false, false }, hard_mask: true);
            h.put(np.array(new[] { -1 }), new long[] { 7 }, "wrap");
            Data(h).Should().Equal(1, 2, 3, 7);
            Mask(h).Should().Equal(true, false, false, false);

            // A bad mode: the lookup's IndexError first when the index is out of range, else the clipmode ValueError.
            h = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { true, false, false, false }, hard_mask: true);
            h.Invoking(v => v.put(np.array(new[] { 9 }), new long[] { 7 }, "bogus")).Should().Throw<IndexError>();
            h.Invoking(v => v.put(np.array(new[] { 1 }), new long[] { 7 }, "bogus")).Should().Throw<ValueError>()
                .WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')");
        }

        /// <summary>
        ///     A masked values array cannot be resized (<c>MaskedArray.resize</c> always raises), so a hard-masked put of
        ///     masked values raises — the masked constant and a masked array without a mask included. Without a mask
        ///     array (nomask) the hard branch is skipped and masked values work.
        /// </summary>
        [TestMethod]
        public void HardMask_MaskedValues_Raise()
        {
            const string text = "A masked array does not own its data and therefore cannot be resized.\nUse the numpy.ma.resize function instead.";
            Hard1D().Invoking(v => v.put(np.array(new[] { 0, 2 }), np.ma.array(new long[] { 7, 8 }, mask: new[] { false, true })))
                .Should().Throw<ValueError>().Where(e => e.Message == text);
            Hard1D().Invoking(v => v.put(np.array(new[] { 0, 2 }), np.ma.array(new long[] { 7, 8 })))
                .Should().Throw<ValueError>().Where(e => e.Message == text);
            Hard1D().Invoking(v => v.put(np.array(new[] { 0, 2 }), np.ma.masked))
                .Should().Throw<ValueError>().Where(e => e.Message == text);
            Hard2D().Invoking(v => v.put(np.array(new[] { 0, 1 }), np.ma.masked))
                .Should().Throw<ValueError>().Where(e => e.Message == text);

            var nomask = np.ma.array(new long[] { 1, 2, 3, 4 }, hard_mask: true);
            nomask.put(np.array(new[] { 0, 2 }), np.ma.array(new long[] { 7, 8 }, mask: new[] { false, true }));
            Data(nomask).Should().Equal(7, 2, 8, 4);
            Mask(nomask).Should().Equal(false, false, true, false);
            nomask = np.ma.array(new long[] { 1, 2, 3, 4 }, hard_mask: true);
            nomask.put(np.array(new[] { 0, 2 }), np.ma.masked);
            Data(nomask).Should().Equal(0, 2, 0, 4);
            Mask(nomask).Should().Equal(true, false, true, false);
        }

        /// <summary>
        ///     An <see cref="NDArray"/> values is the CALLER's array, so <c>ndarray.resize</c>'s reference check refuses a
        ///     size change, and its single-segment and OWNDATA checks come first. A same-size array is relabelled in memory
        ///     order (an F-order array's values go in column-major order, as NumPy's). NumSharp relabels an alias, so the
        ///     caller's array keeps its shape (<c>[Misaligned]</c>: NumPy reshapes it in place).
        /// </summary>
        [TestMethod]
        public void HardMask_CallerArrayValues_FollowNdarrayResize()
        {
            const string refcount = "cannot resize an array that references or is referenced\nby another array in this way.\nUse the np.resize function or refcheck=False";
            Hard1D().Invoking(v => v.put(np.array(new[] { 0, 2, 3 }), np.array(new long[] { 7, 8 })))
                .Should().Throw<IncorrectShapeException>().Where(e => e.Message == refcount);
            Hard1D().Invoking(v => v.put(np.array(new[] { 0, 2 }), NDArray.Scalar(5L)))
                .Should().Throw<IncorrectShapeException>().Where(e => e.Message == refcount);

            var zeros = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new bool[4], hard_mask: true);
            zeros.Invoking(v => v.put(np.array(new[] { 0, 1 }), np.arange(4)["::2"])).Should().Throw<IncorrectShapeException>()
                .WithMessage("resize only works on single-segment arrays");
            zeros.Invoking(v => v.put(np.array(new[] { 0, 1, 2 }), np.arange(4)["0:2"])).Should().Throw<IncorrectShapeException>()
                .WithMessage("cannot resize this array: it does not own its data");

            var f = np.asfortranarray(np.array(new long[,] { { 10, 11 }, { 12, 13 } }));
            zeros = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new bool[4], hard_mask: true);
            zeros.put(np.array(new[] { 0, 1, 2, 3 }), f);
            Data(zeros).Should().Equal(10, 12, 11, 13);
            IsNomask(zeros).Should().BeTrue();      // an all-False mask shrinks to nomask after the put
            f.shape.Should().Equal(2L, 2L);

            var column = np.array(new long[,] { { 7 }, { 8 } });
            var x = Hard1D();
            x.put(np.array(new[] { 0, 2 }), column);
            Data(x).Should().Equal(7, 2, 8, 4);
            column.shape.Should().Equal(2L, 1L);    // NumPy reshapes the caller's array to (2,); NumSharp keeps it
        }

        /// <summary>
        ///     A C# number or array becomes a fresh array FIRST under a hard mask (NumPy's <c>narray(values)</c>), which is
        ///     then cast into the target: a literal out of range wraps there, while the soft branch packs it as NumPy's
        ///     Python scalar and raises the OverflowError.
        /// </summary>
        [TestMethod]
        public void HardMask_ConvertsBeforeTheCast_SoftBranchPacksTheLiteral()
        {
            NDMaskedArray Int8(bool hard) => np.ma.array(np.array(new sbyte[] { 1, 2, 3, 4 }), mask: new[] { false, false, true, false },
                hard_mask: hard);

            var h = Int8(true);
            h.put(np.array(new[] { 0 }), 300);
            h.data.flatten().ToArray<sbyte>().Should().Equal(44, 2, 3, 4);

            var s = Int8(false);
            s.Invoking(v => v.put(np.array(new[] { 0 }), 300)).Should().Throw<OverflowException>()
                .WithMessage("Python integer 300 out of bounds for int8");
            s.data.flatten().ToArray<sbyte>().Should().Equal(1, 2, 3, 4);

            var hi = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { false, true, false, false }, hard_mask: true);
            hi.put(np.array(new[] { 0 }), double.NaN);
            Data(hi)[0].Should().Be(long.MinValue);
            var si = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { false, true, false, false });
            si.Invoking(v => v.put(np.array(new[] { 0 }), double.NaN)).Should().Throw<ValueError>()
                .WithMessage("cannot convert float NaN to integer");
        }

        /// <summary>
        ///     <c>ndarray.put</c> reads a masked values' DATA: the soft branch writes the masked constant's 0.0 (cast) into
        ///     the data under the masked slots, and a plain-array target takes the raw data, masked slots included. The old
        ///     implementation left the data under a masked constant unchanged and skipped the plain target.
        /// </summary>
        [TestMethod]
        public void MaskedValues_WriteTheirData()
        {
            var soft = np.ma.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, mask: new[,] { { false, true, false }, { false, false, true } });
            soft.put(np.array(new[] { 1, 4 }), np.ma.masked);
            Data(soft).Should().Equal(1, 0, 3, 4, 0, 6);
            Mask(soft).Should().Equal(false, true, false, false, true, true);

            soft = np.ma.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, mask: new[,] { { false, true, false }, { false, false, true } });
            soft.put(np.array(new[] { 0, 4 }), np.ma.array(new long[] { 7, 8 }, mask: new[] { false, true }));
            Data(soft).Should().Equal(7, 2, 3, 4, 8, 6);
            Mask(soft).Should().Equal(false, true, false, false, true, true);

            var plain = np.arange(4);
            np.ma.put(plain, np.array(new[] { 0, 2 }), np.ma.masked);
            plain.ToArray<long>().Should().Equal(0, 1, 0, 3);
            plain = np.arange(4);
            np.ma.put(plain, np.array(new[] { 0, 2 }), np.ma.array(new long[] { 7, 8 }, mask: new[] { true, false }));
            plain.ToArray<long>().Should().Equal(7, 1, 8, 3);
            var plainF = np.arange(4.0);
            np.ma.put(plainF, np.array(new[] { 0, 2 }), np.ma.masked);
            plainF.ToArray<double>().Should().Equal(0, 1, 0, 3);
        }

        /// <summary>
        ///     NumPy writes the mask IN PLACE (<c>getmaskarray(self)</c> is the mask itself), so a view sharing it sees the
        ///     change; when the put unmasks every slot, the array's mask shrinks to nomask while the view keeps the
        ///     all-False mask array.
        /// </summary>
        [TestMethod]
        public void SharedMask_IsWrittenInPlace()
        {
            var x = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { false, true, false, false });
            var y = (NDMaskedArray)x[":"];
            x.put(np.array(new[] { 0 }), np.ma.masked);
            Mask(y).Should().Equal(true, true, false, false);
            Data(y).Should().Equal(0, 2, 3, 4);

            x = np.ma.array(new long[] { 1, 2, 3, 4 }, mask: new[] { false, true, false, false });
            y = (NDMaskedArray)x[":"];
            x.put(np.array(new[] { 1 }), 9);
            IsNomask(x).Should().BeTrue();
            IsNomask(y).Should().BeFalse();
            Mask(y).Should().Equal(false, false, false, false);
            Data(y).Should().Equal(1, 9, 3, 4);
        }

        /// <summary>0-d and empty targets: the soft branch is ndarray.put's; the hard branch's lookup raises first.</summary>
        [TestMethod]
        public void ZeroDimensionalAndEmptyTargets()
        {
            var s = np.ma.array(NDArray.Scalar(5L), mask: true);
            s.put(NDArray.Scalar(0L), 9);
            s.data.GetInt64().Should().Be(9);
            IsNomask(s).Should().BeTrue();

            var h = np.ma.array(NDArray.Scalar(5L), mask: false, hard_mask: true);
            h.Invoking(v => v.put(NDArray.Scalar(0L), 9)).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 0-dimensional, but 1 were indexed");

            var emptyHard = np.ma.array(np.zeros(0), mask: np.zeros(0, np.@bool), hard_mask: true);
            emptyHard.Invoking(v => v.put(np.array(new[] { 0 }), new long[] { 1 })).Should().Throw<IndexError>()
                .WithMessage("index 0 is out of bounds for axis 0 with size 0");
            var emptySoft = np.ma.array(np.zeros(0), mask: np.zeros(0, np.@bool));
            emptySoft.Invoking(v => v.put(np.array(new[] { 0 }), new long[] { 1 })).Should().Throw<IndexError>()
                .WithMessage("cannot replace elements of an empty array");

            // An empty float index array is an index the mask lookup refuses.
            Hard1D().Invoking(v => v.put(np.array(new double[0]), new long[] { 1 })).Should().Throw<IndexError>()
                .WithMessage("arrays used as indices must be of integer (or boolean) type");
            var x = Hard1D();
            x.put(np.array(new int[0]), new long[] { 1 });
            Data(x).Should().Equal(1, 2, 3, 4);
        }
    }
}
