using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators;

/// <summary>
/// WRITEMASKED/ARRAYMASK EXECUTION + VIRTUAL operands.
///
/// Every expectation here is pinned to NumPy 2.4.2 output (probed via python_run).
///
/// The load-bearing semantic, verified against NumPy: masking is enforced in
/// exactly ONE place — the buffered copy-back (npyiter_copy_from_buffers,
/// nditer_api.c:2001-2026). An unbuffered WRITEMASKED operand writes the
/// array directly and the mask is the kernel's contract; NumPy 2.x skips
/// buffers for compatible operands even under flags=['buffered'], so the
/// 'buffered' flag alone does NOT imply enforcement.
///
/// VIRTUAL in NumPy 2.x is allocate-equivalent: npyiter_allocate_arrays
/// allocates a real backing array for EVERY null operand (the NEP-12
/// buffer-only semantics never landed; NPY_OP_ITFLAG_VIRTUAL's only consumer
/// is DebugPrint) — but unlike ALLOCATE the requested dtype is DISCARDED
/// (npyiter_prepare_one_operand nulls op_dtype for the non-ALLOCATE branch)
/// and the common dtype of the real operands is used instead.
/// </summary>
[TestClass]
public class NpyIterWriteMaskedExecutionTests
{
    private static NDArray Mask8() =>
        np.array(new[] { true, false, true, false, true, false, true, false });

    private static NpyIterPerOpFlags[] MaskedPair() => new[]
    {
        NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
        NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
    };

    // =====================================================================
    // Masked write execution — where enforcement does and does NOT happen
    // =====================================================================

    [TestMethod]
    public void MaskedWrite_Unbuffered_MaskNotEnforced()
    {
        // NumPy 2.4.2: no buffering -> the iterator exposes direct array
        // pointers; writing every slot writes every slot. [99.]*8
        var a = np.zeros(new Shape(8), np.float64);
        using (var it = NpyIterRef.MultiNew(2, new[] { a, Mask8() },
            NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            MaskedPair()))
        {
            do { it.SetValue(99.0, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(99.0, a.GetDouble(i), $"element {i}: unbuffered writes go straight to the array");
    }

    [TestMethod]
    public void MaskedWrite_BufferedButLinearSameDtype_MaskNotEnforced()
    {
        // NumPy 2.4.2: flags=['buffered'] with contiguous same-dtype operands
        // engages NO buffer (BUFNEVER) -> mask still not enforced. [99.]*8
        // This pins the buffer-when-required parity: 'buffered' is a
        // permission, not a promise.
        var a = np.zeros(new Shape(8), np.float64);
        using (var it = NpyIterRef.MultiNew(2, new[] { a, Mask8() },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            MaskedPair()))
        {
            do { it.SetValue(99.0, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(99.0, a.GetDouble(i), $"element {i}: no buffer engaged, no enforcement (NumPy parity)");
    }

    [TestMethod]
    public void MaskedWrite_BufferedCast_MaskEnforced()
    {
        // NumPy 2.4.2: op_dtypes=['float32'] forces the f64 array through a
        // buffer -> masked copy-back. [99, 0, 99, 0, 99, 0, 99, 0]
        var a = np.zeros(new Shape(8), np.float64);
        using (var it = NpyIterRef.MultiNew(2, new[] { a, Mask8() },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(i % 2 == 0 ? 99.0 : 0.0, a.GetDouble(i), $"element {i}");
    }

    [TestMethod]
    public void MaskedWrite_BufferedCast_ReadWriteIncrement()
    {
        // NumPy 2.4.2: x[...] = x + 10 under mask — copy-IN is full (the
        // kernel sees original values), copy-BACK is masked.
        // arange(8) -> [10, 1, 12, 3, 14, 5, 16, 7]
        var a = np.arange(8).astype(np.float64);
        using (var it = NpyIterRef.MultiNew(2, new[] { a, Mask8() },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(it.GetValue<float>(0) + 10f, 0); } while (it.Iternext());
        }

        var expected = new double[] { 10, 1, 12, 3, 14, 5, 16, 7 };
        for (int i = 0; i < 8; i++)
            Assert.AreEqual(expected[i], a.GetDouble(i), $"element {i}");
    }

    [TestMethod]
    public void MaskedWrite_WriteOnly_UnmaskedSlotsKeepOriginals()
    {
        // NumPy 2.4.2: writeonly+writemasked, full(8, 5.0) target ->
        // [99, 5, 99, 5, 99, 5, 99, 5] (no copy-in, masked copy-back leaves
        // unmasked array slots untouched).
        var b = np.full(new Shape(8), 5.0, np.float64);
        using (var it = NpyIterRef.MultiNew(2, new[] { b, Mask8() },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            new[]
            {
                NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.WRITEMASKED,
                NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
            },
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(i % 2 == 0 ? 99.0 : 5.0, b.GetDouble(i), $"element {i}");
    }

    [TestMethod]
    public void MaskedWrite_StridedMask_MaskItselfBuffered()
    {
        // NumPy 2.4.2 (W5): the mask is a strided view -> non-linear -> the
        // MASK op buffers too; the masked copy-back must read the mask's
        // BUFFER, not its array (nditer_api.c:2009-2014 BUFNEVER switch).
        // [99, 0, 99, 0, 99, 0, 99, 0]
        var a = np.zeros(new Shape(8), np.float64);
        var mbig = np.zeros(new Shape(16), np.bool_);
        mbig["::4"] = np.array(new[] { true, true, true, true });
        var m2 = mbig["::2"]; // [T,F,T,F,T,F,T,F] strided view

        using (var it = NpyIterRef.MultiNew(2, new[] { a, m2 },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(i % 2 == 0 ? 99.0 : 0.0, a.GetDouble(i), $"element {i}");
    }

    [TestMethod]
    public void MaskedWrite_BroadcastMaskRow_Over2D()
    {
        // NumPy 2.4.2 (W6): mask (4,) broadcast over op (2,4) ->
        // ravel [99, 0, 99, 0, 99, 0, 99, 0]
        var a = np.zeros(new Shape(2, 4), np.float64);
        var mrow = np.array(new[] { true, false, true, false });

        using (var it = NpyIterRef.MultiNew(2, new[] { a, mrow },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        var flat = np.ravel(a);
        for (int i = 0; i < 8; i++)
            Assert.AreEqual(i % 2 == 0 ? 99.0 : 0.0, flat.GetDouble(i), $"flat element {i}");
    }

    [TestMethod]
    public void MaskedWrite_MultiWindow_AllWindowsFlushMasked()
    {
        // NumPy 2.4.2 (W7): N=20005 (3 windows: 8192+8192+3621), mask every
        // 3rd element -> 6669 written, 13336 zeros kept, 0 wrong.
        const int N = 20_005;
        var a = np.zeros(new Shape(N), np.float64);
        var mN = np.equal(np.mod(np.arange(N), np.array(3)), np.array(0));

        using (var it = NpyIterRef.MultiNew(2, new[] { a, mN },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        long wrote = 0, kept = 0;
        for (int i = 0; i < N; i++)
        {
            double v = a.GetDouble(i);
            bool expect = i % 3 == 0;
            if (expect && v == 99.0) wrote++;
            else if (!expect && v == 0.0) kept++;
            else Assert.Fail($"element {i}: {v} (masked={expect}) — window-boundary mask drift");
        }
        Assert.AreEqual(6669L, wrote);
        Assert.AreEqual(13336L, kept);
    }

    [TestMethod]
    public void MaskedWrite_Uint8Mask_NonzeroIsTrue()
    {
        // NumPy 2.4.2: "Only bool and uint8 masks are supported." — uint8
        // works, any nonzero byte counts as True. mask {2,0,255,0,1,0,7,0}
        // -> [99, 0, 99, 0, 99, 0, 99, 0]
        var a = np.zeros(new Shape(8), np.float64);
        var m8 = np.array(new byte[] { 2, 0, 255, 0, 1, 0, 7, 0 });

        using (var it = NpyIterRef.MultiNew(2, new[] { a, m8 },
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
            MaskedPair(),
            new[] { NPTypeCode.Single, NPTypeCode.Empty }))
        {
            do { it.SetValue(99.0f, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 8; i++)
            Assert.AreEqual(i % 2 == 0 ? 99.0 : 0.0, a.GetDouble(i), $"element {i}");
    }

    // =====================================================================
    // Validation — NumPy 2.4.2 error texts, matched verbatim
    // =====================================================================

    [TestMethod]
    public void WriteMasked_ReadOnly_ThrowsNumPyText()
    {
        var ex = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { np.zeros(new Shape(8), np.float64), Mask8() },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.WRITEMASKED,
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                }).Dispose());

        Assert.AreEqual(
            "The iterator flag WRITEMASKED may only be used with READWRITE or WRITEONLY",
            ex.Message);
    }

    [TestMethod]
    public void WriteMaskedPairing_ThrowsNumPyTexts()
    {
        var a = np.zeros(new Shape(8), np.float64);
        var m = Mask8();

        var e2 = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, m },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED | NpyIterPerOpFlags.ARRAYMASK,
                    NpyIterPerOpFlags.READONLY,
                }).Dispose());
        Assert.AreEqual("The iterator flag WRITEMASKED may not be used together with ARRAYMASK", e2.Message);

        var e3 = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(3, new[] { a, m, m.copy() },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                }).Dispose());
        Assert.AreEqual("Only one iterator operand may receive an ARRAYMASK flag", e3.Message);

        var e4 = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, m },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
                    NpyIterPerOpFlags.READONLY,
                }).Dispose());
        Assert.AreEqual(
            "An iterator operand was flagged as WRITEMASKED, but no ARRAYMASK operand was given to supply the mask",
            e4.Message);

        var e5 = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, m },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE,
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
                }).Dispose());
        Assert.AreEqual(
            "An iterator operand was flagged as the ARRAYMASK, but no WRITEMASKED operands were given to use the mask",
            e5.Message);
    }

    [TestMethod]
    public void ArrayMask_WouldReduce_ThrowsNumPyText()
    {
        // NumPy 2.4.2: mask (1,) broadcast over (3,4) writemasked rw target,
        // mask itself READWRITE + REDUCE_OK -> the mask reducing would let a
        // True flip to False after a write-back. ValueError, exact text.
        var a = np.zeros(new Shape(3, 4), np.float64);
        var mr = np.array(new[] { true });

        var ex = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, mr },
                NpyIterGlobalFlags.REDUCE_OK, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.ARRAYMASK,
                }).Dispose());

        Assert.AreEqual(
            "output operand requires a reduction, but is flagged as the ARRAYMASK operand " +
            "which is not permitted to be the result of a reduction",
            ex.Message);
    }

    [TestMethod]
    public void WriteMaskedReduce_MaskBroadcastsWider_StandardPath_Throws()
    {
        // NumPy 2.4.2 (E9): op (4,) reducing against mask (3,4) WITHOUT
        // op_axes — the deferred post-stride check must fire on the standard
        // broadcast path too (the op_axes path already had it).
        var asm = np.zeros(new Shape(4), np.float64);
        var mbig = np.ones(new Shape(3, 4), np.bool_);

        var ex = Assert.ThrowsException<InvalidOperationException>(() =>
            NpyIterRef.MultiNew(2, new[] { asm, mbig },
                NpyIterGlobalFlags.REDUCE_OK, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                MaskedPair()).Dispose());

        Assert.AreEqual(
            "Iterator reduction operand is WRITEMASKED, but also broadcasts to multiple " +
            "mask values. There can be only one mask value per WRITEMASKED element.",
            ex.Message);
    }

    [TestMethod]
    public void NonBoolMask_BufferedWrite_ThrowsNumPyText()
    {
        // NumPy 2.4.2 (E7): the masked transfer fn rejects non-bool/uint8
        // masks — TypeError("Only bool and uint8 masks are supported.").
        // Fires when the WRITEMASKED operand actually buffers (cast here).
        var a = np.zeros(new Shape(8), np.float64);
        var m16 = np.ones(new Shape(8), np.int16);

        var ex = Assert.ThrowsException<NotSupportedException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, m16 },
                NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAME_KIND_CASTING,
                MaskedPair(),
                new[] { NPTypeCode.Single, NPTypeCode.Empty }).Dispose());

        Assert.AreEqual("Only bool and uint8 masks are supported.", ex.Message);
    }

    [TestMethod]
    public void NonBoolMask_Unbuffered_NoError()
    {
        // NumPy 2.4.2 (E6): without buffering no masked transfer fn is built,
        // so an int16 mask constructs fine (and the mask is unenforced).
        var a = np.zeros(new Shape(8), np.float64);
        var m16 = np.ones(new Shape(8), np.int16);

        using var it = NpyIterRef.MultiNew(2, new[] { a, m16 },
            NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            MaskedPair());
        Assert.AreEqual(1, it.MaskOp);
    }

    [TestMethod]
    [Misaligned]
    public void BufferedReduce_WriteMasked_WriteBackRefusesLoudly()
    {
        // NumPy 2.4.2 supports masked write-back on buffered reductions (it
        // restricts the reduce double-loop split). NumSharp's legacy
        // buffered-reduce machinery predates masked copy-back; rather than
        // silently writing unmasked slots it refuses at the write-back point.
        // Construction succeeds (NumPy parity); proper support lands with
        // reductions-through-core (roadmap Wave 5).
        var x = np.arange(12).reshape(3, 4).astype(np.int32);
        var mask = np.array(new[] { true, false, true, false });
        var y = np.zeros(new int[] { 4 }, np.int32);

        var opAxes = new[]
        {
            new[] { 0, 1 },
            new[] { -1, 0 },
            new[] { -1, 0 },
        };
        var opFlags = new[]
        {
            NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
            NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
        };

        var ex = Assert.ThrowsException<NotSupportedException>(() =>
        {
            using var it = NpyIterRef.AdvancedNew(
                nop: 3,
                op: new[] { x, mask, y },
                flags: NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: opFlags,
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: opAxes);

            Assert.IsTrue(it.IsReduction, "construction must succeed (NumPy parity)");
            do { it.SetValue(7, 2); } while (it.Iternext());
        });

        StringAssert.Contains(ex.Message, "WRITEMASKED write-back is not supported on a buffered REDUCE");
    }

    // =====================================================================
    // VIRTUAL operands — NumPy 2.x allocate-equivalent semantics
    // =====================================================================

    [TestMethod]
    public void Virtual_NullOp_AllocatesCommonDtype()
    {
        // NumPy 2.4.2 (V5): no dtype request -> common dtype of the real
        // operands (float32 here); a REAL array is allocated and exposed.
        var src = np.arange(6).astype(np.float32);
        var ops = new NDArray[] { null, src };

        using var it = NpyIterRef.MultiNew(2, ops,
            NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL,
                NpyIterPerOpFlags.READONLY,
            });

        Assert.IsNotNull(ops[0], "VIRTUAL allocates a real backing array (NumPy 2.x)");
        Assert.AreEqual(NPTypeCode.Single, ops[0].typecode);
        Assert.AreEqual(6, (int)ops[0].size);
    }

    [TestMethod]
    public void Virtual_RequestedDtype_IsDiscarded()
    {
        // NumPy 2.4.2 (V1): op_dtypes=[int32] on a VIRTUAL operand is
        // DISCARDED (npyiter_prepare_one_operand nulls op_dtype on the
        // non-ALLOCATE branch) -> operands[0].dtype is float32, NOT int32,
        // and the iteration runs cast-free on float32.
        var src = np.arange(6).astype(np.float32);
        var ops = new NDArray[] { null, src };

        using var it = NpyIterRef.MultiNew(2, ops,
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL,
                NpyIterPerOpFlags.READONLY,
            },
            new[] { NPTypeCode.Int32, NPTypeCode.Empty });

        Assert.AreEqual(NPTypeCode.Single, ops[0].typecode,
            "the requested int32 must be discarded in favor of the common dtype");
    }

    [TestMethod]
    public void Virtual_PlusAllocate_HonorsRequestedDtype()
    {
        // NumPy 2.4.2 (V7): when BOTH flags are set, ALLOCATE wins
        // (nditer_constr.c:1009 checks it first) and the request holds.
        var src = np.arange(6).astype(np.float32);
        var ops = new NDArray[] { null, src };

        using var it = NpyIterRef.MultiNew(2, ops,
            NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL | NpyIterPerOpFlags.ALLOCATE,
                NpyIterPerOpFlags.READONLY,
            },
            new[] { NPTypeCode.Int64, NPTypeCode.Empty });

        Assert.AreEqual(NPTypeCode.Int64, ops[0].typecode);
    }

    [TestMethod]
    public void Virtual_WriteThrough_PersistsInAllocatedArray()
    {
        // NumPy 2.4.2 (V2): v[...] = s*2 through the iterator lands in the
        // allocated array. arange(6, f32) -> [0, 2, 4, 6, 8, 10]
        var src = np.arange(6).astype(np.float32);
        var ops = new NDArray[] { null, src };

        using (var it = NpyIterRef.MultiNew(2, ops,
            NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL,
                NpyIterPerOpFlags.READONLY,
            }))
        {
            do { it.SetValue(it.GetValue<float>(1) * 2f, 0); } while (it.Iternext());
        }

        for (int i = 0; i < 6; i++)
            Assert.AreEqual(2f * i, ops[0].GetSingle(i), $"element {i}");
    }

    [TestMethod]
    public void Virtual_WithoutReadwrite_ThrowsNumPyText()
    {
        // NumPy 2.4.2 (V3) — including NumPy's doubled "be", verbatim.
        var src = np.arange(6).astype(np.float32);

        var ex = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new NDArray[] { null, src },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.VIRTUAL,
                    NpyIterPerOpFlags.READONLY,
                }).Dispose());

        Assert.AreEqual("The iterator flag VIRTUAL should be be used together with READWRITE", ex.Message);
    }

    [TestMethod]
    public void Virtual_OnNonNullOperand_ThrowsNumPyText()
    {
        // NumPy 2.4.2 (V4).
        var a = np.arange(6).astype(np.float32);

        var ex = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new[] { a, a.copy() },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL,
                    NpyIterPerOpFlags.READONLY,
                }).Dispose());

        Assert.AreEqual("Iterator operand flag VIRTUAL was specified, but the operand was not NULL", ex.Message);
    }

    [TestMethod]
    public void NullOperand_NeitherAllocateNorVirtual_ThrowsNumPyText()
    {
        // NumPy 2.4.2 (V6).
        var src = np.arange(6).astype(np.float32);

        var ex = Assert.ThrowsException<ArgumentException>(() =>
            NpyIterRef.MultiNew(2, new NDArray[] { null, src },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READWRITE,
                    NpyIterPerOpFlags.READONLY,
                }).Dispose());

        Assert.AreEqual("Iterator operand was NULL, but neither the ALLOCATE nor the VIRTUAL flag was specified", ex.Message);
    }

    [TestMethod]
    public void Virtual_ArrayMask_DefaultsToBool()
    {
        // NumPy nditer_constr.c:1041-1049: a null ARRAYMASK operand with no
        // dtype request defaults to bool (not the common dtype).
        var a = np.zeros(new Shape(4), np.float64);
        var ops = new NDArray[] { a, null };

        using var it = NpyIterRef.MultiNew(2, ops,
            NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.WRITEMASKED,
                NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.VIRTUAL | NpyIterPerOpFlags.ARRAYMASK,
            });

        Assert.AreEqual(NPTypeCode.Boolean, ops[1].typecode);
        Assert.AreEqual(1, it.MaskOp);
    }
}
