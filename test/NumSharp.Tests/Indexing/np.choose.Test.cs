using System;
using System.Numerics;

namespace NumSharp.Tests.Indexing;

/// <summary>
/// Tests for <c>np.choose(a, choices, out, mode)</c> and the <c>ndarray.choose</c> method — a port of
/// NumPy 2.4.2's <c>PyArray_Choose</c> (item_selection.c). Every expected value / dtype / error was
/// taken from running NumPy 2.4.2 on win-amd64. The exhaustive bit-exact value/layout coverage was
/// validated out-of-band by a 1,485-case differential replay (11 dtypes × ranks 1-3 × 5 index layouts
/// × 3 choice layouts × 3 modes, all bit-identical); these tests pin the API surface, the NEP50 dtype
/// resolution, the error contract, and the harder broadcast/layout/out edges.
/// </summary>
[TestClass]
public class ChooseTests
{
    // ---- basics + modes ------------------------------------------------------

    [TestMethod]
    public void Choose_Basic()
    {
        var choices = new NDArray[]
        {
            np.array(new[] { 0, 1, 2, 3 }), np.array(new[] { 10, 11, 12, 13 }),
            np.array(new[] { 20, 21, 22, 23 }), np.array(new[] { 30, 31, 32, 33 }),
        };
        var r = np.choose(np.array(new[] { 2, 3, 1, 0 }), choices);
        r.GetTypeCode.Should().Be(NPTypeCode.Int32);
        r.ToArray<int>().Should().Equal(20, 31, 12, 3);
    }

    [TestMethod]
    public void Choose_Clip()
    {
        var choices = new NDArray[]
        {
            np.array(new[] { 0, 1, 2, 3 }), np.array(new[] { 10, 11, 12, 13 }),
            np.array(new[] { 20, 21, 22, 23 }), np.array(new[] { 30, 31, 32, 33 }),
        };
        // 4 clips to n-1 = 3; negatives clip to 0.
        np.choose(np.array(new[] { 2, 4, 1, 0 }), choices, mode: "clip").ToArray<int>().Should().Equal(20, 31, 12, 3);
        np.choose(np.array(new[] { -5, -2, 1, 0 }), choices, mode: "clip").ToArray<int>().Should().Equal(0, 1, 12, 3);
    }

    [TestMethod]
    public void Choose_Wrap()
    {
        var choices = new NDArray[]
        {
            np.array(new[] { 0, 1, 2, 3 }), np.array(new[] { 10, 11, 12, 13 }),
            np.array(new[] { 20, 21, 22, 23 }), np.array(new[] { 30, 31, 32, 33 }),
        };
        // 4 wraps to 4 mod 4 = 0.
        np.choose(np.array(new[] { 2, 4, 1, 0 }), choices, mode: "wrap").ToArray<int>().Should().Equal(20, 1, 12, 3);
        // negatives wrap Python-style: -1→3, -2→2.
        np.choose(np.array(new[] { -1, -2, 1, 0 }), choices, mode: "wrap").ToArray<int>().Should().Equal(30, 21, 12, 3);
    }

    [TestMethod]
    public void Choose_Wrap_LargeMagnitude()
    {
        var choices = new NDArray[] { np.array(new[] { 1, 1, 1 }), np.array(new[] { 2, 2, 2 }), np.array(new[] { 3, 3, 3 }) };
        // 100 % 3 = 1, -100 wraps to 2, 7 % 3 = 1.
        np.choose(np.array(new[] { 100, -100, 7 }), choices, mode: "wrap").ToArray<int>().Should().Equal(2, 3, 2);
    }

    // ---- NEP50 dtype resolution ---------------------------------------------

    [TestMethod]
    public void Choose_ScalarChoices_WeakInt_Int64()
    {
        // np.choose([[1,0,1],...], [-10, 10]) → int64 (two weak python ints → default int).
        var a = np.array(new[,] { { 1, 0, 1 }, { 0, 1, 0 } });
        var r = np.choose(a, new object[] { -10, 10 });
        r.GetTypeCode.Should().Be(NPTypeCode.Int64);
        r.flatten().ToArray<long>().Should().Equal(10, -10, 10, -10, 10, -10);
    }

    [TestMethod]
    public void Choose_WeakInt_AdoptsStrong_AndWraps()
    {
        // int8 choice + weak 1000 → int8, 1000 wraps to -24.
        var r = np.choose(np.array(new[] { 0, 1 }), new object[] { np.array(new sbyte[] { 1, 2 }), 1000 });
        r.GetTypeCode.Should().Be(NPTypeCode.SByte);
        r.ToArray<sbyte>().Should().Equal(1, -24);
    }

    [TestMethod]
    public void Choose_WeakFloat_ForcesDouble()
    {
        var r = np.choose(np.array(new[] { 0, 1 }), new object[] { np.array(new sbyte[] { 1, 2 }), 1.5 });
        r.GetTypeCode.Should().Be(NPTypeCode.Double);
        r.ToArray<double>().Should().Equal(1.0, 1.5);
    }

    [TestMethod]
    public void Choose_Dtype_Uint8_Int8_PromotesInt16()
    {
        var r = np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new byte[] { 1, 2 }), np.array(new sbyte[] { 3, 4 }) });
        r.GetTypeCode.Should().Be(NPTypeCode.Int16);
        r.ToArray<short>().Should().Equal(1, 4);
    }

    [TestMethod]
    public void Choose_Dtype_BoolChoices_StayBool()
    {
        var r = np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new[] { true, false }), np.array(new[] { false, true }) });
        r.GetTypeCode.Should().Be(NPTypeCode.Boolean);
        r.ToArray<bool>().Should().Equal(true, true);
    }

    [TestMethod]
    public void Choose_Dtype_Complex()
    {
        var r = np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new Complex[] { new Complex(1, 2), 2 }), np.array(new Complex[] { 3, 4 }) });
        r.GetTypeCode.Should().Be(NPTypeCode.Complex);
        r.ToArray<Complex>().Should().Equal(new Complex(1, 2), new Complex(4, 0));
    }

    [TestMethod]
    public void Choose_IndexDtype_IsIrrelevantToResultDtype()
    {
        // int8 index, int8 choices → int8 (the index dtype never influences the result dtype).
        var r = np.choose(np.array(new sbyte[] { 0, 1 }),
            new NDArray[] { np.array(new sbyte[] { 1, 2 }), np.array(new sbyte[] { 3, 4 }) });
        r.GetTypeCode.Should().Be(NPTypeCode.SByte);
        r.ToArray<sbyte>().Should().Equal(1, 4);
    }

    // ---- broadcasting --------------------------------------------------------

    [TestMethod]
    public void Choose_Broadcast_ThreeWay()
    {
        // a(2,1,1), c1(1,3,1), c2(1,1,5) → (2,3,5); res[0]=c1 broadcast, res[1]=c2 broadcast.
        var a = np.arange(2).reshape(2, 1, 1);
        var c1 = np.array(new[] { 1, 2, 3 }).reshape(1, 3, 1);
        var c2 = np.array(new[] { -1, -2, -3, -4, -5 }).reshape(1, 1, 5);
        var r = np.choose(a, new object[] { c1, c2 });
        r.shape.Should().Equal(2, 3, 5);
        r.GetInt32(0, 2, 4).Should().Be(3);   // c1 broadcast: [0,2,:] = 3
        r.GetInt32(1, 0, 4).Should().Be(-5);  // c2 broadcast: [1,:,4] = -5
    }

    [TestMethod]
    public void Choose_Broadcast_ScalarChoice()
    {
        var r = np.choose(np.array(new[] { 0, 1, 0, 1 }),
            new NDArray[] { NDArray.Scalar(5), np.array(new[] { 10, 11, 12, 13 }) });
        r.ToArray<int>().Should().Equal(5, 11, 5, 13);
    }

    [TestMethod]
    public void Choose_ScalarIndex_BroadcastsToChoiceShape()
    {
        // A 0-d index against (3,) choices broadcasts to (3,): the whole chosen array is returned.
        var r = np.choose(np.array(1), new NDArray[] { np.array(new[] { 10, 20, 30 }), np.array(new[] { 40, 50, 60 }) });
        r.shape.Should().Equal(3);
        r.ToArray<int>().Should().Equal(40, 50, 60);
    }

    [TestMethod]
    public void Choose_ScalarIndex_ScalarChoices_ReturnsZeroD()
    {
        // A 0-d index against 0-d (scalar) choices stays 0-d.
        var r = np.choose(np.array(1), new NDArray[] { NDArray.Scalar(10), NDArray.Scalar(20) });
        r.ndim.Should().Be(0);
        r.GetInt32().Should().Be(20);
    }

    // ---- choices forms -------------------------------------------------------

    [TestMethod]
    public void Choose_SingleNdarray_OutermostAxisIsSequence()
    {
        var c = np.array(new[,] { { 0, 1, 2, 3 }, { 10, 11, 12, 13 }, { 20, 21, 22, 23 }, { 30, 31, 32, 33 } });
        np.choose(np.array(new[] { 2, 3, 1, 0 }), c).ToArray<int>().Should().Equal(20, 31, 12, 3);
    }

    [TestMethod]
    public void Choose_SingleChoice_N1()
    {
        np.choose(np.array(new[] { 0, 0, 0 }), new NDArray[] { np.array(new[] { 5, 6, 7 }) }).ToArray<int>().Should().Equal(5, 6, 7);
    }

    [TestMethod]
    public void Choose_InstanceMethod_MatchesFunction()
    {
        var a = np.array(new[] { 1, 0, 1 });
        var chs = new NDArray[] { np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }) };
        a.choose(chs).ToArray<int>().Should().Equal(np.choose(a, chs).ToArray<int>());
    }

    // ---- layouts (the strided kernel) ---------------------------------------

    [TestMethod]
    public void Choose_NegativeStrideIndex()
    {
        var idx = np.array(new[] { 0, 1, 0, 1 })["::-1"]; // [1,0,1,0]
        var r = np.choose(idx, new NDArray[] { np.array(new[] { 10, 11, 12, 13 }), np.array(new[] { 20, 21, 22, 23 }) });
        r.ToArray<int>().Should().Equal(20, 11, 22, 13);
    }

    [TestMethod]
    public void Choose_TransposedChoices()
    {
        // Choices transposed; index C-contiguous — both broadcast to the same (2,2) shape.
        var idx = np.array(new[,] { { 0, 1 }, { 1, 0 } });
        var c0 = np.array(new[,] { { 1, 2 }, { 3, 4 } }).T;   // [[1,3],[2,4]]
        var c1 = np.array(new[,] { { 5, 6 }, { 7, 8 } }).T;   // [[5,7],[6,8]]
        var r = np.choose(idx, new NDArray[] { c0, c1 });
        // r[0,0]=c0[0,0]=1, r[0,1]=c1[0,1]=7, r[1,0]=c1[1,0]=6, r[1,1]=c0[1,1]=4
        r.flatten().ToArray<int>().Should().Equal(1, 7, 6, 4);
    }

    // ---- out= ----------------------------------------------------------------

    [TestMethod]
    public void Choose_Out_SameDtype_ReturnsInstance()
    {
        var o = np.zeros(new Shape(2), NPTypeCode.Int64);
        var r = np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new long[] { 1, 2 }), np.array(new long[] { 3, 4 }) }, @out: o);
        ReferenceEquals(r, o).Should().BeTrue();
        o.ToArray<long>().Should().Equal(1, 4);
    }

    [TestMethod]
    public void Choose_Out_DifferentDtype_UnsafeCast()
    {
        // float choices into an int32 out — truncates toward zero (unsafe cast).
        var o = np.zeros(new Shape(2), NPTypeCode.Int32);
        np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new[] { 1.5, 2.5 }), np.array(new[] { 3.5, 4.5 }) }, @out: o);
        o.ToArray<int>().Should().Equal(1, 4);
    }

    [TestMethod]
    public void Choose_Out_StridedView_WritesThrough()
    {
        var idx = np.array(new[,] { { 0, 1 }, { 1, 0 }, { 0, 1 } }); // (3,2)
        var c0 = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var c1 = np.array(new[,] { { 10, 20 }, { 30, 40 }, { 50, 60 } });
        var outBase = np.zeros(new Shape(2, 3), NPTypeCode.Int32);
        var outT = outBase.T; // (3,2) non-contiguous
        var r = np.choose(idx, new NDArray[] { c0, c1 }, @out: outT);
        ReferenceEquals(r, outT).Should().BeTrue();
        outT.flatten().ToArray<int>().Should().Equal(1, 20, 30, 4, 5, 60);
        outBase.flatten().ToArray<int>().Should().Equal(1, 30, 5, 20, 4, 60); // write-through
    }

    [TestMethod]
    public void Choose_Out_Raise_LeavesOutUnchangedOnError()
    {
        var o = np.array(new[] { -7, -7 });
        Action act = () => np.choose(np.array(new[] { 0, 2 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) }, @out: o);
        act.Should().Throw<ValueError>();
        o.ToArray<int>().Should().Equal(-7, -7); // temp-then-copy guarantees no partial write
    }

    // ---- 16-byte + Half widths ----------------------------------------------

    [TestMethod]
    public void Choose_Decimal()
    {
        var d0 = np.array(new[] { 1m, 2m, 3m, 4m });
        var d1 = np.array(new[] { 10m, 20m, 30m, 40m });
        var d2 = np.array(new[] { 100m, 200m, 300m, 400m });
        np.choose(np.array(new[] { 2, 0, 1, 2 }), new NDArray[] { d0, d1, d2 })
            .ToArray<decimal>().Should().Equal(100m, 2m, 30m, 400m);
    }

    [TestMethod]
    public void Choose_Half()
    {
        var c0 = np.array(new[] { (Half)1, (Half)2, (Half)3 });
        var c1 = np.array(new[] { (Half)10, (Half)20, (Half)30 });
        var r = np.choose(np.array(new[] { 0, 1, 0 }), new NDArray[] { c0, c1 });
        r.GetTypeCode.Should().Be(NPTypeCode.Half);
        r.ToArray<Half>().Should().Equal((Half)1, (Half)20, (Half)3);
    }

    // ---- empty ---------------------------------------------------------------

    [TestMethod]
    public void Choose_EmptyIndex_ReturnsEmpty()
    {
        var r = np.choose(np.array(new int[] { }),
            new NDArray[] { np.array(new double[] { }), np.array(new double[] { }) });
        r.size.Should().Be(0);
        r.GetTypeCode.Should().Be(NPTypeCode.Double);
    }

    // ---- error contract ------------------------------------------------------

    [TestMethod]
    public void Choose_Raise_OutOfBounds_Throws()
    {
        Action act = () => np.choose(np.array(new[] { 0, 2 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) });
        act.Should().Throw<ValueError>().WithMessage("invalid entry in choice array");
    }

    [TestMethod]
    public void Choose_EmptyChoices_Throws()
    {
        Action act = () => np.choose(np.array(new[] { 0, 1 }), new NDArray[] { });
        act.Should().Throw<ValueError>().WithMessage("0-length sequence.");
    }

    [TestMethod]
    public void Choose_FloatIndex_Throws()
    {
        Action act = () => np.choose(np.array(new[] { 0.0, 1.0 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) });
        act.Should().Throw<TypeError>()
            .WithMessage("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
    }

    [TestMethod]
    public void Choose_UInt64Index_Throws()
    {
        Action act = () => np.choose(np.array(new ulong[] { 0, 1 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) });
        act.Should().Throw<TypeError>()
            .WithMessage("Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule 'safe'");
    }

    [TestMethod]
    public void Choose_Out_WrongShape_Throws()
    {
        Action act = () => np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) },
            @out: np.zeros(new Shape(3), NPTypeCode.Int32));
        act.Should().Throw<TypeError>().WithMessage("choose: invalid shape for output array.");
    }

    [TestMethod]
    public void Choose_Out_ReadOnly_Throws()
    {
        // A broadcast view is non-writeable and (2,) matches the result shape → the writeable check fires.
        var ro = np.broadcast_to(np.array(0), new Shape(2));
        Action act = () => np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) }, @out: ro);
        act.Should().Throw<ValueError>().WithMessage("output array is read-only");
    }

    [TestMethod]
    public void Choose_Mode_Unknown_Throws()
    {
        Action act = () => np.choose(np.array(new[] { 0, 1 }),
            new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) }, mode: "foo");
        act.Should().Throw<ValueError>().WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'foo')");
    }

    [TestMethod]
    public void Choose_Mode_NearMiss_Throws()
    {
        // First char matches but the spelling/case is not exact → the second clip-mode message.
        foreach (var bad in new[] { "CLIP", "r", "Wrap", "c" })
        {
            Action act = () => np.choose(np.array(new[] { 0, 1 }),
                new NDArray[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) }, mode: bad);
            act.Should().Throw<ValueError>().WithMessage("Use one of 'clip', 'raise', or 'wrap' for clip mode");
        }
    }

    [TestMethod]
    public void Choose_SingleZeroDChoices_Throws()
    {
        Action act = () => np.choose(np.array(0), np.array(5));
        act.Should().Throw<TypeError>().WithMessage("iteration over a 0-d array");
    }
}
