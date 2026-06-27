using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NPY_ITFLAG_TRANSFERFLAGS_SHIFT packing.
    /// NumPy: nditer_api.c:903 (NDIter_GetTransferFlags), nditer_constr.c:3542 (packing).
    ///
    /// Semantics: Combined NPY_ARRAYMETHOD_FLAGS from all transfer functions are packed
    /// into the top 8 bits of ItFlags at construction. GetTransferFlags shifts them back out.
    ///
    /// In .NET, REQUIRES_PYAPI is never set (no Python). SUPPORTS_UNALIGNED and
    /// NO_FLOATINGPOINT_ERRORS are always set (raw byte pointer casts, silent truncation).
    /// </summary>
    [TestClass]
    public class NDIterTransferFlagsTests
    {
        [TestMethod]
        public void TransferFlags_NoCast_ReturnsBasicFlags()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NDIterRef.New(a);

            var flags = it.GetTransferFlags();
            // Same-type copy: SUPPORTS_UNALIGNED + NO_FLOATINGPOINT_ERRORS + IS_REORDERABLE
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.SUPPORTS_UNALIGNED));
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS));
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.IS_REORDERABLE));
            Assert.IsFalse(flags.HasFlag(NDArrayMethodFlags.REQUIRES_PYAPI), "REQUIRES_PYAPI should never be set in .NET");
        }

        [TestMethod]
        public void TransferFlags_Cast_Int32ToFloat64_ReturnsAllFlags()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NDIterRef.AdvancedNew(
                nop: 1,
                op: new[] { a },
                flags: NDIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NDIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double });

            var flags = it.GetTransferFlags();
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.SUPPORTS_UNALIGNED));
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS));
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.IS_REORDERABLE));
            Assert.IsFalse(flags.HasFlag(NDArrayMethodFlags.REQUIRES_PYAPI));
        }

        [TestMethod]
        public void TransferFlags_NeverSetsPyApi()
        {
            // Exercise several safe casts — none should set REQUIRES_PYAPI in .NET.
            // Per NumPy np.can_cast(src, dst, 'safe'):
            var casts = new[]
            {
                (src: NPTypeCode.Int32, dst: NPTypeCode.Double),     // int32→float64: safe
                (src: NPTypeCode.Int16, dst: NPTypeCode.Int32),      // int16→int32: safe
                (src: NPTypeCode.Single, dst: NPTypeCode.Double),    // float32→float64: safe
                (src: NPTypeCode.Boolean, dst: NPTypeCode.Int32),    // bool→int32: safe
            };

            foreach (var (src, dst) in casts)
            {
                var a = np.arange(4).astype(src);
                using var it = NDIterRef.AdvancedNew(
                    nop: 1,
                    op: new[] { a },
                    flags: NDIterGlobalFlags.BUFFERED,
                    order: NPY_ORDER.NPY_KEEPORDER,
                    casting: NPY_CASTING.NPY_SAFE_CASTING,
                    opFlags: new[] { NDIterPerOpFlags.READONLY },
                    opDtypes: new[] { dst });

                var flags = it.GetTransferFlags();
                Assert.IsFalse(flags.HasFlag(NDArrayMethodFlags.REQUIRES_PYAPI),
                    $"Cast {src}→{dst} should not set REQUIRES_PYAPI");
            }
        }

        [TestMethod]
        public void TransferFlags_Shift_IsAt24()
        {
            // Packing happens at bit 24. Verify roundtrip.
            Assert.AreEqual(24, NDIterConstants.TRANSFERFLAGS_SHIFT);
            Assert.AreEqual(0xFF000000u, NDIterConstants.TRANSFERFLAGS_MASK);
        }

        [TestMethod]
        public void TransferFlags_RuntimeFlags_Mask()
        {
            // NPY_METH_RUNTIME_FLAGS == REQUIRES_PYAPI | NO_FLOATINGPOINT_ERRORS
            // Matches NumPy dtype_api.h:96.
            Assert.AreEqual(
                NDArrayMethodFlags.REQUIRES_PYAPI | NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS,
                NDArrayMethodFlags.RUNTIME_FLAGS);
        }

        [TestMethod]
        public void TransferFlags_MultiOperand_Combined()
        {
            var x = np.arange(5).astype(np.int32);
            var y = np.zeros(new int[] { 5 }, np.int32);

            using var it = NDIterRef.MultiNew(
                nop: 2,
                op: new[] { x, y },
                flags: NDIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });

            var flags = it.GetTransferFlags();
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.SUPPORTS_UNALIGNED));
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS));
        }

        [TestMethod]
        public void TransferFlags_DoNotCollideWithOtherItFlags()
        {
            // Top 8 bits are reserved for transfer flags. Other flags should
            // not bleed into them.
            var a = np.arange(10).reshape(2, 5).astype(np.int32);
            using var it = NDIterRef.AdvancedNew(
                nop: 1,
                op: new[] { a },
                flags: NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NDIterPerOpFlags.READONLY });

            // Both standard flags (HasIndex, HasMultiIndex) AND transfer flags should be readable
            Assert.IsTrue(it.HasIndex, "C_INDEX should set HASINDEX");
            Assert.IsTrue(it.HasMultiIndex, "MULTI_INDEX should set HASMULTIINDEX");

            var flags = it.GetTransferFlags();
            Assert.IsTrue(flags.HasFlag(NDArrayMethodFlags.SUPPORTS_UNALIGNED));
        }

        [TestMethod]
        public void TransferFlags_AccessibleAfterIteration()
        {
            // Transfer flags must remain intact during iteration
            var a = np.arange(5).astype(np.int32);
            using var it = NDIterRef.AdvancedNew(
                nop: 1,
                op: new[] { a },
                flags: NDIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NDIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double });

            var flagsBefore = it.GetTransferFlags();
            do { var _ = it.GetValue<double>(0); } while (it.Iternext());
            var flagsAfter = it.GetTransferFlags();

            Assert.AreEqual(flagsBefore, flagsAfter, "Transfer flags should not change during iteration");
        }
    }
}
