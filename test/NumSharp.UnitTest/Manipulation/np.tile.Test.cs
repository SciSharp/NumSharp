using System;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Battletest for <c>np.tile</c>. Expected values verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class TileTests
    {
        // ----------------------------------------------------------------------
        // Section 1 — params int[] overload, NumPy doc examples
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_1D_Reps2_Repeats()
        {
            // NumPy: np.tile([0,1,2], 2) → [0,1,2,0,1,2]
            var got = np.tile(np.arange(3), 2);
            got.shape.Should().Equal(6L);
            for (int i = 0; i < 6; i++)
                ((long)got[i]).Should().Be(i % 3);
        }

        [TestMethod]
        public void Tile_1D_Reps_2_2_PromotesAxis()
        {
            // NumPy: np.tile([0,1,2], (2,2)) → shape (2,6)
            var got = np.tile(np.arange(3), 2, 2);
            got.shape.Should().Equal(new long[] { 2, 6 });
            int[] expected = { 0, 1, 2, 0, 1, 2 };
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 6; j++)
                    ((long)got[i, j]).Should().Be(expected[j]);
        }

        [TestMethod]
        public void Tile_1D_Reps_2_1_2_Promotes3D()
        {
            // NumPy: np.tile([0,1,2], (2,1,2)) → shape (2,1,6)
            var got = np.tile(np.arange(3), 2, 1, 2);
            got.shape.Should().Equal(new long[] { 2, 1, 6 });
            int[] expected = { 0, 1, 2, 0, 1, 2 };
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 6; j++)
                    ((long)got[i, 0, j]).Should().Be(expected[j]);
        }

        [TestMethod]
        public void Tile_2D_Reps2_PromotesRepsTo_1_2()
        {
            // NumPy: np.tile([[1,2],[3,4]], 2) → shape (2,4) (reps promoted to (1,2))
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 2);
            got.shape.Should().Equal(new long[] { 2, 4 });
            int[,] expected = { { 1, 2, 1, 2 }, { 3, 4, 3, 4 } };
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    ((int)got[i, j]).Should().Be(expected[i, j]);
        }

        [TestMethod]
        public void Tile_2D_Reps_2_1()
        {
            // NumPy: np.tile([[1,2],[3,4]], (2,1)) → shape (4,2)
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 2, 1);
            got.shape.Should().Equal(new long[] { 4, 2 });
            int[,] expected = { { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 } };
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++)
                    ((int)got[i, j]).Should().Be(expected[i, j]);
        }

        [TestMethod]
        public void Tile_1D_Vertical_4_1()
        {
            // NumPy: np.tile([1,2,3,4], (4,1)) → shape (4,4) — vertical stack
            var c = np.array(new[] { 1, 2, 3, 4 });
            var got = np.tile(c, 4, 1);
            got.shape.Should().Equal(new long[] { 4, 4 });
            int[] row = { 1, 2, 3, 4 };
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    ((int)got[i, j]).Should().Be(row[j]);
        }

        [TestMethod]
        public void Tile_2D_Reps_2_3_FullExpansion()
        {
            // NumPy: np.tile([[1,2],[3,4]], (2,3)) → (4,6)
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 2, 3);
            got.shape.Should().Equal(new long[] { 4, 6 });
            int[,] expected =
            {
                { 1, 2, 1, 2, 1, 2 },
                { 3, 4, 3, 4, 3, 4 },
                { 1, 2, 1, 2, 1, 2 },
                { 3, 4, 3, 4, 3, 4 },
            };
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 6; j++)
                    ((int)got[i, j]).Should().Be(expected[i, j]);
        }

        // ----------------------------------------------------------------------
        // Section 2 — Edge cases
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_Scalar_Reps3()
        {
            // NumPy: np.tile(5, 3) → [5,5,5]
            var got = np.tile(np.array(5), 3);
            got.shape.Should().Equal(3L);
            for (int i = 0; i < 3; i++)
                ((int)got[i]).Should().Be(5);
        }

        [TestMethod]
        public void Tile_Scalar_Reps_2_3()
        {
            // NumPy: np.tile(5, (2,3)) → [[5,5,5],[5,5,5]]
            var got = np.tile(np.array(5), 2, 3);
            got.shape.Should().Equal(new long[] { 2, 3 });
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    ((int)got[i, j]).Should().Be(5);
        }

        [TestMethod]
        public void Tile_Empty_Reps3_ProducesEmpty()
        {
            // NumPy: np.tile([], 3) → array([], shape=(0,))
            var got = np.tile(np.array(new int[] { }), 3);
            got.shape.Should().Equal(0L);
        }

        [TestMethod]
        public void Tile_Reps0_ProducesEmpty()
        {
            // NumPy: np.tile([1,2,3], 0) → array([])
            var got = np.tile(np.array(new[] { 1, 2, 3 }), 0);
            got.shape.Should().Equal(0L);
        }

        [TestMethod]
        public void Tile_Reps1_ReturnsCopy()
        {
            // NumPy: np.tile(arr, 1) returns a copy (not a view).
            var src = np.array(new[] { 1, 2, 3 });
            var got = np.tile(src, 1);
            got.shape.Should().Equal(3L);

            // Mutating the result must not affect the source.
            got[0] = 99;
            ((int)src[0]).Should().Be(1);
        }

        [TestMethod]
        public void Tile_AllOnes_2D_ReturnsCopy()
        {
            var src = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(src, 1, 1);
            got.shape.Should().Equal(new long[] { 2, 2 });
            got[0, 0] = 99;
            ((int)src[0, 0]).Should().Be(1);
        }

        // ----------------------------------------------------------------------
        // Section 3 — 3D
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_3D_Reps_2_1_3()
        {
            // NumPy: np.tile(arange(8).reshape(2,2,2), (2,1,3)) → shape (4,2,6)
            var a = np.arange(8).reshape((2, 2, 2));
            var got = np.tile(a, 2, 1, 3);
            got.shape.Should().Equal(new long[] { 4, 2, 6 });
            // Spot-check values against NumPy output
            ((long)got[0, 0, 0]).Should().Be(0);
            ((long)got[0, 0, 1]).Should().Be(1);
            ((long)got[0, 0, 5]).Should().Be(1);
            ((long)got[2, 0, 0]).Should().Be(0); // axis-0 wrap-around
            ((long)got[3, 1, 5]).Should().Be(7);
        }

        // ----------------------------------------------------------------------
        // Section 4 — Dtype preservation
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_DtypePreserved_Int32()
        {
            var got = np.tile(np.array(new[] { 1, 2, 3 }).astype(np.int32), 2);
            got.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Tile_DtypePreserved_Float32()
        {
            var got = np.tile(np.array(new[] { 1f, 2f, 3f }), 2);
            got.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Tile_DtypePreserved_Bool()
        {
            var got = np.tile(np.array(new[] { true, false }), 3);
            got.dtype.Should().Be(typeof(bool));
            bool[] expected = { true, false, true, false, true, false };
            for (int i = 0; i < 6; i++)
                ((bool)got[i]).Should().Be(expected[i]);
        }

        // ----------------------------------------------------------------------
        // Section 5 — Layout
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_OutputIsCContiguous()
        {
            var got = np.tile(np.array(new[] { 1, 2, 3 }), 2);
            got.Shape.IsContiguous.Should().BeTrue();
        }

        // ----------------------------------------------------------------------
        // Section 6 — Validation
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_NegativeReps_Throws()
        {
            Action act = () => np.tile(np.array(new[] { 1, 2, 3 }), -1);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Tile_NullArray_Throws()
        {
            Action act = () => np.tile(null!, 2);
            act.Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void Tile_NullReps_Throws()
        {
            Action act = () => np.tile(np.array(new[] { 1, 2, 3 }), (int[])null!);
            act.Should().Throw<ArgumentNullException>();
        }

        // ----------------------------------------------------------------------
        // Section 7 — Long overload
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_LongRepsOverload()
        {
            var got = np.tile(np.array(new[] { 1, 2, 3 }), new long[] { 2L });
            got.shape.Should().Equal(6L);
        }

        // ----------------------------------------------------------------------
        // Section 8 — Empty reps (NumPy: np.tile(a, ()) returns a copy of a)
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_NoReps_ReturnsCopyOfOriginalShape()
        {
            // NumPy: np.tile(np.array([1,2,3]), ()) → array([1,2,3]), shape (3,)
            var src = np.array(new[] { 1, 2, 3 });
            var got = np.tile(src);
            got.shape.Should().Equal(3L);
            for (int i = 0; i < 3; i++) ((int)got[i]).Should().Be(i + 1);
            // Must be a copy (writable, independent of src).
            got[0] = 99;
            ((int)src[0]).Should().Be(1);
        }

        [TestMethod]
        public void Tile_NoReps_PreservesNDim()
        {
            // NumPy: np.tile(2d_array, ()) → preserves 2D shape
            var src = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(src);
            got.shape.Should().Equal(new long[] { 2, 2 });
        }

        // ----------------------------------------------------------------------
        // Section 9 — Non-contiguous / strided / broadcast / sliced input
        // Tile must materialize data correctly regardless of input memory layout.
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_TransposedInput_Reps2()
        {
            // a = arange(6).reshape(2,3).T → shape (3,2), non-contiguous
            // NumPy np.tile(a, 2) →
            //   [[0 3 0 3]
            //    [1 4 1 4]
            //    [2 5 2 5]]
            var a = np.arange(6).reshape((2, 3)).T;
            a.Shape.IsContiguous.Should().BeFalse();
            var got = np.tile(a, 2);
            got.shape.Should().Equal(new long[] { 3, 4 });
            long[,] expected = { { 0, 3, 0, 3 }, { 1, 4, 1, 4 }, { 2, 5, 2, 5 } };
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((long)got[i, j]).Should().Be(expected[i, j]);
        }

        [TestMethod]
        public void Tile_TransposedInput_Reps_2_2()
        {
            var a = np.arange(6).reshape((2, 3)).T;
            var got = np.tile(a, 2, 2);
            got.shape.Should().Equal(new long[] { 6, 4 });
            long[,] expected = {
                { 0, 3, 0, 3 }, { 1, 4, 1, 4 }, { 2, 5, 2, 5 },
                { 0, 3, 0, 3 }, { 1, 4, 1, 4 }, { 2, 5, 2, 5 },
            };
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 4; j++)
                    ((long)got[i, j]).Should().Be(expected[i, j]);
        }

        [TestMethod]
        public void Tile_BroadcastedInput_Reps2()
        {
            // b = broadcast_to(arange(3), (2,3)) → shape (2,3), stride=0 on axis 0
            // NumPy np.tile(b, 2) →
            //   [[0 1 2 0 1 2]
            //    [0 1 2 0 1 2]]
            var b = np.broadcast_to(np.arange(3), new Shape(2, 3));
            b.Shape.IsBroadcasted.Should().BeTrue();
            var got = np.tile(b, 2);
            got.shape.Should().Equal(new long[] { 2, 6 });
            long[] row = { 0, 1, 2, 0, 1, 2 };
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 6; j++)
                    ((long)got[i, j]).Should().Be(row[j]);
            // Output must be writable even though input was a read-only broadcast view.
            got.Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void Tile_SlicedInput_Reps3()
        {
            // c = arange(10)[::2] → [0,2,4,6,8], non-contiguous
            var c = np.arange(10)["::2"];
            c.Shape.IsContiguous.Should().BeFalse();
            var got = np.tile(c, 3);
            got.shape.Should().Equal(15L);
            long[] expected = { 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8 };
            for (int i = 0; i < 15; i++) ((long)got[i]).Should().Be(expected[i]);
        }

        // ----------------------------------------------------------------------
        // Section 10 — reps with zeros at various positions
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_ZeroReps_LeadingAxis()
        {
            // NumPy: np.tile([[1,2],[3,4]], (0,2)) → shape (0,4)
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 0, 2);
            got.shape.Should().Equal(new long[] { 0, 4 });
            got.size.Should().Be(0);
            got.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Tile_ZeroReps_TrailingAxis()
        {
            // NumPy: np.tile([[1,2],[3,4]], (2,0)) → shape (4,0)
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 2, 0);
            got.shape.Should().Equal(new long[] { 4, 0 });
            got.size.Should().Be(0);
        }

        // ----------------------------------------------------------------------
        // Section 11 — A.ndim > len(reps): reps promoted by prepending 1s
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_2D_With_Scalar_Reps_TilesLastAxis()
        {
            // NumPy: np.tile([[1,2],[3,4]], 3) → reps promoted to (1,3) → shape (2,6)
            var b = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var got = np.tile(b, 3);
            got.shape.Should().Equal(new long[] { 2, 6 });
            int[,] expected = { { 1, 2, 1, 2, 1, 2 }, { 3, 4, 3, 4, 3, 4 } };
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 6; j++)
                    ((int)got[i, j]).Should().Be(expected[i, j]);
        }

        [TestMethod]
        public void Tile_3D_With_Scalar_Reps_TilesLastAxis()
        {
            // NumPy: np.tile(arange(24).reshape(2,3,4), (2,)) →
            //   reps promoted to (1,1,2) → shape (2,3,8)
            var a = np.arange(24).reshape((2, 3, 4));
            var got = np.tile(a, 2);
            got.shape.Should().Equal(new long[] { 2, 3, 8 });
            // Spot-check: got[0,0,:] = [0,1,2,3,0,1,2,3]
            long[] row0 = { 0, 1, 2, 3, 0, 1, 2, 3 };
            for (int j = 0; j < 8; j++) ((long)got[0, 0, j]).Should().Be(row0[j]);
            // got[1,2,:] = [20,21,22,23,20,21,22,23]
            long[] rowLast = { 20, 21, 22, 23, 20, 21, 22, 23 };
            for (int j = 0; j < 8; j++) ((long)got[1, 2, j]).Should().Be(rowLast[j]);
        }

        // ----------------------------------------------------------------------
        // Section 12 — 4D
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_4D_Reps_2_1_3_1()
        {
            // NumPy: np.tile(arange(24).reshape(2,3,2,2), (2,1,3,1)) → shape (4,3,6,2)
            // Axis 2 has dim 2 tiled 3x → 6. Pattern on axis 2: [a0,a1,a0,a1,a0,a1].
            var a = np.arange(24).reshape((2, 3, 2, 2));
            var got = np.tile(a, 2, 1, 3, 1);
            got.shape.Should().Equal(new long[] { 4, 3, 6, 2 });
            // got[0,0,:,:] = tile [[0,1],[2,3]] along axis 0 three times
            long[,] block00 = {
                { 0, 1 }, { 2, 3 }, { 0, 1 }, { 2, 3 }, { 0, 1 }, { 2, 3 }
            };
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 2; j++)
                    ((long)got[0, 0, i, j]).Should().Be(block00[i, j]);
            // got[2,0,:,:] = got[0,0,:,:] (axis 0 tile)
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 2; j++)
                    ((long)got[2, 0, i, j]).Should().Be(block00[i, j]);
        }

        // ----------------------------------------------------------------------
        // Section 13 — Dtype coverage across all 12 NumSharp types
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_AllDtypes_PreservedAndCorrect()
        {
            // Repeat [1,2,3] twice → [1,2,3,1,2,3] across every dtype.

            var byteGot = np.tile(np.array(new byte[] { 1, 2, 3 }), 2);
            byteGot.dtype.Should().Be(typeof(byte));
            byteGot.shape.Should().Equal(6L);
            byte[] byteExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((byte)byteGot[i]).Should().Be(byteExp[i]);

            var shortGot = np.tile(np.array(new short[] { 1, 2, 3 }), 2);
            shortGot.dtype.Should().Be(typeof(short));
            short[] shortExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((short)shortGot[i]).Should().Be(shortExp[i]);

            var ushortGot = np.tile(np.array(new ushort[] { 1, 2, 3 }), 2);
            ushortGot.dtype.Should().Be(typeof(ushort));
            ushort[] ushortExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((ushort)ushortGot[i]).Should().Be(ushortExp[i]);

            var intGot = np.tile(np.array(new int[] { 1, 2, 3 }), 2);
            intGot.dtype.Should().Be(typeof(int));
            int[] intExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((int)intGot[i]).Should().Be(intExp[i]);

            var uintGot = np.tile(np.array(new uint[] { 1, 2, 3 }), 2);
            uintGot.dtype.Should().Be(typeof(uint));
            uint[] uintExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((uint)uintGot[i]).Should().Be(uintExp[i]);

            var longGot = np.tile(np.array(new long[] { 1, 2, 3 }), 2);
            longGot.dtype.Should().Be(typeof(long));
            long[] longExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((long)longGot[i]).Should().Be(longExp[i]);

            var ulongGot = np.tile(np.array(new ulong[] { 1, 2, 3 }), 2);
            ulongGot.dtype.Should().Be(typeof(ulong));
            ulong[] ulongExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((ulong)ulongGot[i]).Should().Be(ulongExp[i]);

            var floatGot = np.tile(np.array(new float[] { 1, 2, 3 }), 2);
            floatGot.dtype.Should().Be(typeof(float));
            float[] floatExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((float)floatGot[i]).Should().Be(floatExp[i]);

            var doubleGot = np.tile(np.array(new double[] { 1, 2, 3 }), 2);
            doubleGot.dtype.Should().Be(typeof(double));
            double[] doubleExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((double)doubleGot[i]).Should().Be(doubleExp[i]);

            var decimalGot = np.tile(np.array(new decimal[] { 1, 2, 3 }), 2);
            decimalGot.dtype.Should().Be(typeof(decimal));
            decimal[] decimalExp = { 1, 2, 3, 1, 2, 3 };
            for (int i = 0; i < 6; i++) ((decimal)decimalGot[i]).Should().Be(decimalExp[i]);

            // Bool — semantics differ (not a number), so verify separately.
            var boolGot = np.tile(np.array(new[] { true, false, true }), 2);
            boolGot.dtype.Should().Be(typeof(bool));
            bool[] boolExpected = { true, false, true, true, false, true };
            for (int i = 0; i < 6; i++) ((bool)boolGot[i]).Should().Be(boolExpected[i]);

            // Char — stores ordinal values.
            var charGot = np.tile(np.array(new[] { 'a', 'b', 'c' }), 2);
            charGot.dtype.Should().Be(typeof(char));
            char[] charExpected = { 'a', 'b', 'c', 'a', 'b', 'c' };
            for (int i = 0; i < 6; i++) ((char)charGot[i]).Should().Be(charExpected[i]);
        }

        // ----------------------------------------------------------------------
        // Section 14 — Scalar 0-d array with higher-dim reps
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_Scalar_Reps_2_1_3()
        {
            // NumPy: np.tile(np.array(7), (2,1,3)) → shape (2,1,3) filled with 7
            var got = np.tile(np.array(7), 2, 1, 3);
            got.shape.Should().Equal(new long[] { 2, 1, 3 });
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    ((int)got[i, 0, j]).Should().Be(7);
        }

        // ----------------------------------------------------------------------
        // Section 15 — reps_len > A.ndim (prepend size-1 axes to A)
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_1D_Reps_2_1_3_4()
        {
            // NumPy: np.tile([1,2,3], (2,1,3,4)) → shape (2,1,3,12)
            var got = np.tile(np.array(new[] { 1, 2, 3 }), 2, 1, 3, 4);
            got.shape.Should().Equal(new long[] { 2, 1, 3, 12 });
            // got[0,0,0,:] = [1,2,3,1,2,3,1,2,3,1,2,3]
            int[] row = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 };
            for (int j = 0; j < 12; j++) ((int)got[0, 0, 0, j]).Should().Be(row[j]);
        }

        // ----------------------------------------------------------------------
        // Section 16 — Independence of source after tile
        // ----------------------------------------------------------------------

        [TestMethod]
        public void Tile_Output_IsIndependentCopy()
        {
            var src = np.array(new[] { 1, 2, 3 });
            var got = np.tile(src, 3);
            got[0] = 100;
            got[3] = 200; // second tile start
            ((int)src[0]).Should().Be(1);
            ((int)src[1]).Should().Be(2);
            ((int)src[2]).Should().Be(3);
        }
    }
}
