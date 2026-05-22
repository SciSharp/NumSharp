using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for the np.* index-conversion family — flat↔multi-coord folding +
/// grid coordinate generation. All three are IL-kernel-backed:
/// <list type="bullet">
///   <item><c>np.unravel_index</c> — divmod-chain per element via UnravelIndexKernel,
///         skip-last-divmod optimization.</item>
///   <item><c>np.ravel_multi_index</c> — mul-add per element + per-axis raise/wrap/clip
///         dispatch via RavelMultiIndexKernel.</item>
///   <item><c>np.indices</c> — slab-tile SIMD memset via IndicesKernel, no per-element
///         divmod.</item>
/// </list>
///
/// Test buckets:
/// <list type="bullet">
///   <item>NumPy parity on each documented example.</item>
///   <item>0-d / 1-D / N-D shape preservation across input/output.</item>
///   <item>Empty arrays — must return empty results without crashing.</item>
///   <item>OOB validation (raise mode + unravel_index out-of-range).</item>
///   <item>Wrap mode with neg/pos overflow including multi-period values.</item>
///   <item>Cross-validation: unravel ∘ ravel and ravel ∘ unravel round-trip identity
///         on random inputs.</item>
/// </list>
/// </summary>
[TestClass]
public class IndexConversionTests
{
    private static long[] AsLongs(NDArray<long> nd)
    {
        var buf = new long[nd.size];
        for (long i = 0; i < nd.size; i++) buf[i] = nd.GetAtIndex(i);
        return buf;
    }

    // =================================================================
    // np.unravel_index
    // =================================================================

    [TestMethod]
    public void UnravelIndex_1D_C_NumPyParity()
    {
        var idx = np.array(new int[] { 22, 41, 37 });
        var res = np.unravel_index(idx, new[] { 7, 6 });
        res.Length.Should().Be(2);
        AsLongs(res[0]).Should().Equal(3L, 6L, 6L);
        AsLongs(res[1]).Should().Equal(4L, 5L, 1L);
    }

    [TestMethod]
    public void UnravelIndex_1D_F_NumPyParity()
    {
        var idx = np.array(new int[] { 22, 41, 37 });
        var res = np.unravel_index(idx, new[] { 7, 6 }, 'F');
        res.Length.Should().Be(2);
        AsLongs(res[0]).Should().Equal(1L, 6L, 2L);
        AsLongs(res[1]).Should().Equal(3L, 5L, 5L);
    }

    [TestMethod]
    public void UnravelIndex_Scalar_4D_NumPyParity()
    {
        var coords = np.unravel_index(1621L, new[] { 6, 7, 8, 9 });
        coords.Should().Equal(3L, 1L, 4L, 1L);
    }

    [TestMethod]
    public void UnravelIndex_Scalar_F_NumPyParity()
    {
        // For shape (7, 6) F-order, flat index 22 → unravel = (1, 3)
        // Because 22 = 1 + 3*7
        var coords = np.unravel_index(22L, new[] { 7, 6 }, 'F');
        coords.Should().Equal(1L, 3L);
    }

    [TestMethod]
    public void UnravelIndex_2D_InputPreservesShape()
    {
        // 2-D input → 2-D output arrays preserving the input's shape.
        var idx = np.array(new int[,] { { 22, 41 }, { 37, 0 } });
        var res = np.unravel_index(idx, new[] { 7, 6 });

        res[0].shape.Should().Equal(2, 2);
        res[1].shape.Should().Equal(2, 2);
        AsLongs(res[0]).Should().Equal(3L, 6L, 6L, 0L);
        AsLongs(res[1]).Should().Equal(4L, 5L, 1L, 0L);
    }

    [TestMethod]
    public void UnravelIndex_0d_Input_Returns0dResults()
    {
        var nd = NDArray.Scalar(1621);
        var res = np.unravel_index(nd, new[] { 6, 7, 8, 9 });
        res.Length.Should().Be(4);
        for (int d = 0; d < 4; d++)
            res[d].ndim.Should().Be(0, $"axis {d} should preserve 0-d shape");
        res[0].GetAtIndex(0).Should().Be(3L);
        res[1].GetAtIndex(0).Should().Be(1L);
        res[2].GetAtIndex(0).Should().Be(4L);
        res[3].GetAtIndex(0).Should().Be(1L);
    }

    [TestMethod]
    public void UnravelIndex_Empty_ReturnsEmptyTuple()
    {
        var idx = np.array(new int[0]);
        var res = np.unravel_index(idx, new[] { 7, 6 });
        res.Length.Should().Be(2);
        res[0].size.Should().Be(0);
        res[1].size.Should().Be(0);
    }

    [TestMethod]
    public void UnravelIndex_OOB_PositiveThrows()
    {
        var idx = np.array(new int[] { 50 });
        var act = () => np.unravel_index(idx, new[] { 7, 6 });
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*index 50*size 42*");
    }

    [TestMethod]
    public void UnravelIndex_OOB_NegativeThrows()
    {
        var idx = np.array(new int[] { -1 });
        var act = () => np.unravel_index(idx, new[] { 7, 6 });
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*index -1*size 42*");
    }

    [TestMethod]
    public void UnravelIndex_Scalar_OOBThrows()
    {
        var act = () => np.unravel_index(42L, new[] { 7, 6 });  // unravel size is 42 → idx 42 OOB
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public void UnravelIndex_InvalidOrder_Throws()
    {
        var act = () => np.unravel_index(0L, new[] { 7, 6 }, 'X');
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void UnravelIndex_5D_StackAllocBoundary()
    {
        // 5-D input forces the kernel's dimStrides stackalloc to exercise > 4 entries.
        var coords = np.unravel_index(123L, new[] { 2, 3, 4, 5, 6 });
        // 123 = 0*360 + 1*120 + 0*30 + 0*6 + 3
        coords.Should().Equal(0L, 1L, 0L, 0L, 3L);
    }

    [TestMethod]
    public void UnravelIndex_NonInt64DtypeInput_CastsAndWorks()
    {
        // Inputs of any integer dtype get cast to int64 internally.
        var idx16 = np.array(new short[] { 22, 41, 37 });
        var idx8 = np.array(new byte[] { 22, 41, 37 });

        AsLongs(np.unravel_index(idx16, new[] { 7, 6 })[0]).Should().Equal(3L, 6L, 6L);
        AsLongs(np.unravel_index(idx8, new[] { 7, 6 })[1]).Should().Equal(4L, 5L, 1L);
    }

    // =================================================================
    // np.ravel_multi_index
    // =================================================================

    [TestMethod]
    public void RavelMultiIndex_Basic_NumPyParity()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 3, 6, 6 }),
            np.array(new int[] { 4, 5, 1 })
        };
        var res = np.ravel_multi_index(coords, new[] { 7, 6 });
        AsLongs(res).Should().Equal(22L, 41L, 37L);
    }

    [TestMethod]
    public void RavelMultiIndex_FOrder_NumPyParity()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 3, 6, 6 }),
            np.array(new int[] { 4, 5, 1 })
        };
        var res = np.ravel_multi_index(coords, new[] { 7, 6 }, "raise", 'F');
        AsLongs(res).Should().Equal(31L, 41L, 13L);
    }

    [TestMethod]
    public void RavelMultiIndex_Clip_NumPyParity()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 3, 6, 6 }),
            np.array(new int[] { 4, 5, 1 })
        };
        var res = np.ravel_multi_index(coords, new[] { 4, 6 }, "clip");
        AsLongs(res).Should().Equal(22L, 23L, 19L);
    }

    [TestMethod]
    public void RavelMultiIndex_PerAxisModes_NumPyParity()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 3, 6, 6 }),
            np.array(new int[] { 4, 5, 1 })
        };
        var res = np.ravel_multi_index(coords, new[] { 4, 4 }, new[] { "clip", "wrap" });
        AsLongs(res).Should().Equal(12L, 13L, 13L);
    }

    [TestMethod]
    public void RavelMultiIndex_Scalar_NumPyParity()
    {
        var v = np.ravel_multi_index(new long[] { 3, 1, 4, 1 }, new[] { 6, 7, 8, 9 });
        v.Should().Be(1621);
    }

    [TestMethod]
    public void RavelMultiIndex_OOB_Raise_Throws()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 8 }),
            np.array(new int[] { 0 })
        };
        var act = () => np.ravel_multi_index(coords, new[] { 7, 6 });
        act.Should().Throw<ArgumentException>().WithMessage("*invalid entry*");
    }

    [TestMethod]
    public void RavelMultiIndex_Wrap_NegativeMultiPeriod()
    {
        // Coord -1 with mode='wrap' in (4, 4) → (3, 3) → flat 15.
        var coords = new NDArray[]
        {
            np.array(new int[] { -1 }),
            np.array(new int[] { -1 })
        };
        var res = np.ravel_multi_index(coords, new[] { 4, 4 }, "wrap");
        AsLongs(res).Should().Equal(15L);
    }

    [TestMethod]
    public void RavelMultiIndex_Wrap_MultiPeriodPositive()
    {
        // Coord 11 in dim 4 with wrap: 11 - 4 = 7 ≥ 4 → fallback to %: 11 % 4 = 3.
        var coords = new NDArray[]
        {
            np.array(new int[] { 11 }),
            np.array(new int[] { 11 })
        };
        var res = np.ravel_multi_index(coords, new[] { 4, 4 }, "wrap");
        // (3, 3) → 15
        AsLongs(res).Should().Equal(15L);
    }

    [TestMethod]
    public void RavelMultiIndex_Wrap_NegativeMultiPeriodLarge()
    {
        // Coord -9 in dim 4 → +4 = -5 (still neg) → % gives -1 → +4 = 3
        var coords = new NDArray[]
        {
            np.array(new int[] { -9 }),
            np.array(new int[] { 0 })
        };
        var res = np.ravel_multi_index(coords, new[] { 4, 4 }, "wrap");
        // (3, 0) → 12
        AsLongs(res).Should().Equal(12L);
    }

    [TestMethod]
    public void RavelMultiIndex_2DCoordArrays_ShapePreserved()
    {
        var coords = new NDArray[]
        {
            np.array(new int[,] { { 3, 6 }, { 6, 0 } }),
            np.array(new int[,] { { 4, 5 }, { 1, 0 } })
        };
        var res = np.ravel_multi_index(coords, new[] { 7, 6 });
        res.shape.Should().Equal(2, 2);
        AsLongs(res).Should().Equal(22L, 41L, 37L, 0L);
    }

    [TestMethod]
    public void RavelMultiIndex_Empty_ReturnsEmpty()
    {
        var coords = new NDArray[]
        {
            np.array(new int[0]),
            np.array(new int[0])
        };
        var res = np.ravel_multi_index(coords, new[] { 7, 6 });
        res.size.Should().Be(0);
    }

    [TestMethod]
    public void RavelMultiIndex_MismatchedCoordShapes_Throws()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 0, 1 }),
            np.array(new int[] { 0, 1, 2 })   // different length
        };
        var act = () => np.ravel_multi_index(coords, new[] { 7, 6 });
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void RavelMultiIndex_DimLengthMismatch_Throws()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 0 }),
        };
        var act = () => np.ravel_multi_index(coords, new[] { 7, 6 });
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void RavelMultiIndex_InvalidMode_Throws()
    {
        var coords = new NDArray[]
        {
            np.array(new int[] { 0 }),
            np.array(new int[] { 0 })
        };
        var act = () => np.ravel_multi_index(coords, new[] { 7, 6 }, "explode");
        act.Should().Throw<ArgumentException>();
    }

    // =================================================================
    // np.indices
    // =================================================================

    [TestMethod]
    public void Indices_2D_DenseShape()
    {
        var i = np.indices(new[] { 2, 3 });
        i.shape.Should().Equal(2, 2, 3);
        i.typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Indices_2D_ValuesMatchNumPy()
    {
        var i = np.indices(new[] { 2, 3 });
        // i[0] = [[0,0,0],[1,1,1]]
        i.GetInt64(0, 0, 0).Should().Be(0);
        i.GetInt64(0, 0, 1).Should().Be(0);
        i.GetInt64(0, 0, 2).Should().Be(0);
        i.GetInt64(0, 1, 0).Should().Be(1);
        i.GetInt64(0, 1, 1).Should().Be(1);
        i.GetInt64(0, 1, 2).Should().Be(1);

        // i[1] = [[0,1,2],[0,1,2]]
        i.GetInt64(1, 0, 0).Should().Be(0);
        i.GetInt64(1, 0, 1).Should().Be(1);
        i.GetInt64(1, 0, 2).Should().Be(2);
        i.GetInt64(1, 1, 0).Should().Be(0);
        i.GetInt64(1, 1, 1).Should().Be(1);
        i.GetInt64(1, 1, 2).Should().Be(2);
    }

    [TestMethod]
    public void Indices_1D()
    {
        var i = np.indices(new[] { 5 });
        i.shape.Should().Equal(1, 5);
        for (int k = 0; k < 5; k++)
            i.GetInt64(0, k).Should().Be(k);
    }

    [TestMethod]
    public void Indices_3D_ValuesMatchExpected()
    {
        var i = np.indices(new[] { 2, 3, 4 });
        i.shape.Should().Equal(3, 2, 3, 4);
        // i[d, i0, i1, i2] == iX where X = d
        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 3; i1++)
                for (int i2 = 0; i2 < 4; i2++)
                {
                    i.GetInt64(0, i0, i1, i2).Should().Be(i0);
                    i.GetInt64(1, i0, i1, i2).Should().Be(i1);
                    i.GetInt64(2, i0, i1, i2).Should().Be(i2);
                }
    }

    [TestMethod]
    public void Indices_4D_StackAllocBoundary()
    {
        // 4-D forces the kernel's tile/value loops to exercise multiple levels of nesting.
        var i = np.indices(new[] { 2, 2, 3, 2 });
        i.shape.Should().Equal(4, 2, 2, 3, 2);

        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 2; i1++)
                for (int i2 = 0; i2 < 3; i2++)
                    for (int i3 = 0; i3 < 2; i3++)
                    {
                        i.GetInt64(0, i0, i1, i2, i3).Should().Be(i0);
                        i.GetInt64(1, i0, i1, i2, i3).Should().Be(i1);
                        i.GetInt64(2, i0, i1, i2, i3).Should().Be(i2);
                        i.GetInt64(3, i0, i1, i2, i3).Should().Be(i3);
                    }
    }

    [TestMethod]
    public void Indices_EmptyDimsTuple_Returns1DEmpty()
    {
        var i = np.indices(new int[0]);
        i.shape.Should().Equal(0);
    }

    [TestMethod]
    public void Indices_ZeroDim_ReturnsEmpty()
    {
        var i = np.indices(new[] { 0, 3 });
        i.shape.Should().Equal(2, 0, 3);
        i.size.Should().Be(0);
    }

    [TestMethod]
    public void Indices_Sparse_2D()
    {
        var s = np.indices_sparse(new[] { 2, 3 });
        s.Length.Should().Be(2);
        s[0].shape.Should().Equal(2, 1);
        s[1].shape.Should().Equal(1, 3);

        s[0].GetInt64(0, 0).Should().Be(0);
        s[0].GetInt64(1, 0).Should().Be(1);

        s[1].GetInt64(0, 0).Should().Be(0);
        s[1].GetInt64(0, 1).Should().Be(1);
        s[1].GetInt64(0, 2).Should().Be(2);
    }

    [TestMethod]
    public void Indices_DoubleDtype_CastsCorrectly()
    {
        var i = np.indices(new[] { 2, 3 }, NPTypeCode.Double);
        i.dtype.Should().Be(typeof(double));
        i.GetDouble(1, 1, 2).Should().Be(2.0);
    }

    [TestMethod]
    public void Indices_NegativeDim_Throws()
    {
        var act = () => np.indices(new[] { -1, 3 });
        act.Should().Throw<ArgumentException>();
    }

    // =================================================================
    // Round-trip cross-validation
    // =================================================================

    [TestMethod]
    public void RoundTrip_RavelThenUnravel_RestoresCoords()
    {
        // Coord arrays in (7, 6, 5) shape; ravel then unravel must recover them.
        int d0 = 7, d1 = 6, d2 = 5;
        var rng = new Random(12345);
        int n = 50;
        var c0 = new int[n]; var c1 = new int[n]; var c2 = new int[n];
        for (int i = 0; i < n; i++)
        {
            c0[i] = rng.Next(d0);
            c1[i] = rng.Next(d1);
            c2[i] = rng.Next(d2);
        }

        var coords = new NDArray[] { np.array(c0), np.array(c1), np.array(c2) };
        var flat = np.ravel_multi_index(coords, new[] { d0, d1, d2 });

        var unravelled = np.unravel_index(flat, new[] { d0, d1, d2 });
        unravelled.Length.Should().Be(3);
        for (long i = 0; i < n; i++)
        {
            unravelled[0].GetAtIndex(i).Should().Be(c0[i]);
            unravelled[1].GetAtIndex(i).Should().Be(c1[i]);
            unravelled[2].GetAtIndex(i).Should().Be(c2[i]);
        }
    }

    [TestMethod]
    public void RoundTrip_UnravelThenRavel_RestoresIndex()
    {
        int d0 = 7, d1 = 6, d2 = 5;
        long unravelSize = d0 * d1 * d2;
        var rng = new Random(54321);
        int n = 50;
        var flatVals = new long[n];
        for (int i = 0; i < n; i++) flatVals[i] = rng.NextInt64(0, unravelSize);

        var flat = np.array(flatVals);
        var coords = np.unravel_index(flat, new[] { d0, d1, d2 });
        var roundtripped = np.ravel_multi_index(
            new NDArray[] { coords[0], coords[1], coords[2] },
            new[] { d0, d1, d2 });

        AsLongs(roundtripped).Should().Equal(flatVals);
    }

    [TestMethod]
    public void RoundTrip_FOrder_RavelUnravelConsistent()
    {
        // F-order round-trip: same flat indices must produce coords that re-ravel to themselves.
        int d0 = 5, d1 = 4, d2 = 3;
        long unravelSize = d0 * d1 * d2;
        var rng = new Random(99);
        var flatVals = new long[20];
        for (int i = 0; i < 20; i++) flatVals[i] = rng.NextInt64(0, unravelSize);

        var coords = np.unravel_index(np.array(flatVals), new[] { d0, d1, d2 }, 'F');
        var roundtrip = np.ravel_multi_index(
            new NDArray[] { coords[0], coords[1], coords[2] },
            new[] { d0, d1, d2 }, "raise", 'F');

        AsLongs(roundtrip).Should().Equal(flatVals);
    }
}
