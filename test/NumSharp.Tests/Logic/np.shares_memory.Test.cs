using System;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for <c>np.shares_memory</c> and <c>np.may_share_memory</c> — the memory-overlap
    /// predicate family. All expected values probed against NumPy 2.4.2.
    ///
    /// <para><c>shares_memory</c> answers EXACTLY (default <c>max_work=-1</c>) and raises on the
    /// undecided/overflow outcomes; <c>may_share_memory</c> checks byte bounds only (default
    /// <c>max_work=0</c>) and folds "undecided" into a conservative <c>true</c>. The two diverge on
    /// interleaved views of one buffer — overlapping bounds, no common byte.</para>
    /// </summary>
    [TestClass]
    public class np_shares_memory_Test
    {
        // ---------------------------------------------------------------- shared views ----

        [TestMethod]
        public void SharesMemory_ViewSharesWithBase()
        {
            // >>> a = np.arange(12); b = a[::2]
            // >>> np.shares_memory(a, b), np.may_share_memory(a, b) -> (True, True)
            var a = np.arange(12);
            var b = a["::2"];
            Assert.IsTrue(np.shares_memory(a, b));
            Assert.IsTrue(np.may_share_memory(a, b));
        }

        [TestMethod]
        public void SharesMemory_CopyDoesNotShare()
        {
            // >>> c = a.copy(); np.shares_memory(a, c) -> False ; np.may_share_memory(a, c) -> False
            var a = np.arange(12);
            var c = a.copy();
            Assert.IsFalse(np.shares_memory(a, c));
            Assert.IsFalse(np.may_share_memory(a, c));
        }

        [TestMethod]
        public void SharesMemory_SelfSharesWithSelf()
        {
            // >>> np.shares_memory(a, a) -> True ; np.may_share_memory(a, a) -> True
            var a = np.arange(12);
            Assert.IsTrue(np.shares_memory(a, a));
            Assert.IsTrue(np.may_share_memory(a, a));
        }

        // ---------------------------------------------------- exact-vs-bounds divergence ----

        [TestMethod]
        public void InterleavedViews_MayShareTrue_SharesFalse()
        {
            // The defining difference: even/odd strided views of ONE base have overlapping byte
            // bounds but never a common byte.
            // >>> base = np.arange(20); even = base[::2]; odd = base[1::2]
            // >>> np.may_share_memory(even, odd) -> True
            // >>> np.shares_memory(even, odd)    -> False
            var baseArr = np.arange(20);
            var even = baseArr["::2"];
            var odd = baseArr["1::2"];
            Assert.IsTrue(np.may_share_memory(even, odd));
            Assert.IsFalse(np.shares_memory(even, odd));
            // -2 and a positive budget still reach the exact "False" here.
            Assert.IsFalse(np.shares_memory(even, odd, max_work: -2));
            Assert.IsFalse(np.shares_memory(even, odd, max_work: 1));
        }

        // -------------------------------------------------------------------- max_work ----

        [TestMethod]
        public void SharesMemory_MaxWorkZero_UndecidedThrowsTooHard()
        {
            // A bounds-only budget on a genuinely overlapping pair cannot decide exactly, so
            // shares_memory raises TooHardError (NumPy: "Exceeded max_work").
            var a = np.arange(12);
            var b = a["::2"];
            var ex = Assert.ThrowsException<TooHardError>(() => np.shares_memory(a, b, max_work: 0));
            Assert.AreEqual("Exceeded max_work", ex.Message);
        }

        [TestMethod]
        public void MayShareMemory_MaxWorkZero_IsConservativeTrue()
        {
            // may_share_memory never raises: the undecided outcome becomes True.
            var a = np.arange(12);
            var b = a["::2"];
            Assert.IsTrue(np.may_share_memory(a, b, max_work: 0));
            // Non-overlapping bounds are still a definite False even at the cheap budget.
            Assert.IsFalse(np.may_share_memory(a, a.copy(), max_work: 0));
        }

        [TestMethod]
        public void TooHardError_IsRuntimeError()
        {
            // Callers written against NumPy's hierarchy catch it as RuntimeError.
            var a = np.arange(12);
            var b = a["::2"];
            try
            {
                np.shares_memory(a, b, max_work: 0);
                Assert.Fail("expected TooHardError");
            }
            catch (RuntimeError ex)
            {
                Assert.IsInstanceOfType(ex, typeof(TooHardError));
            }
        }

        // ---------------------------------------------------------------------- errors ----

        [TestMethod]
        public void MaxWorkBelowMinusTwo_ThrowsValueError()
        {
            // >>> np.shares_memory(a, b, max_work=-3) -> ValueError("Invalid value for max_work")
            var a = np.arange(12);
            var b = a["::2"];
            var e1 = Assert.ThrowsException<ValueError>(() => np.shares_memory(a, b, max_work: -3));
            StringAssert.Contains(e1.Message, "Invalid value for max_work");
            var e2 = Assert.ThrowsException<ValueError>(() => np.may_share_memory(a, b, max_work: -3));
            StringAssert.Contains(e2.Message, "Invalid value for max_work");
        }

        // ------------------------------------------------------------- degenerate inputs ----

        [TestMethod]
        public void EmptyArrays_DoNotShare()
        {
            // Zero-size arrays occupy no bytes, so they never overlap (matches NumPy).
            var a = np.arange(12);
            Assert.IsFalse(np.shares_memory(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0, 3))));
            Assert.IsFalse(np.shares_memory(a, a["0:0"]));
            Assert.IsFalse(np.may_share_memory(a, a["0:0"]));
        }

        [TestMethod]
        public void NullInputs_DoNotShare()
        {
            // A null operand has no memory; NumPy coerces None into a fresh distinct array (also False).
            var a = np.arange(12);
            Assert.IsFalse(np.shares_memory(null, a));
            Assert.IsFalse(np.shares_memory(a, null));
            Assert.IsFalse(np.shares_memory(null, null));
            Assert.IsFalse(np.may_share_memory(null, a));
            Assert.IsFalse(np.may_share_memory(a, null));
        }

        // ------------------------------------------------------------- multi-dim layouts ----

        [TestMethod]
        public void TransposeAndColumnViews_Share()
        {
            // A transpose and a column slice are views over the same buffer.
            var m = np.arange(12).reshape(3, 4);
            Assert.IsTrue(np.shares_memory(m, m.T));
            Assert.IsTrue(np.shares_memory(m, m[":, 1"]));
        }

        [TestMethod]
        public void DisjointColumns_ExactFalse_BoundsTrue()
        {
            // Columns 0 and 2 of a C-contiguous matrix never touch the same byte, yet their bounds
            // (first row to last) overlap — the exact/bounds split again.
            var m = np.arange(12).reshape(3, 4);
            Assert.IsFalse(np.shares_memory(m[":, 0"], m[":, 2"]));
            Assert.IsTrue(np.may_share_memory(m[":, 0"], m[":, 2"]));
        }

        [TestMethod]
        public void NegativeStrideView_SharesWithBase()
        {
            // A reversed view aliases the same storage in descending order.
            var a = np.arange(12);
            var rev = a["::-1"];
            Assert.IsTrue(np.shares_memory(a, rev));
            Assert.IsTrue(np.may_share_memory(a, rev));
        }

        // ------------------------------------------------------------------ return type ----

        [TestMethod]
        public void ReturnTypesAreBool()
        {
            var a = np.arange(4);
            bool s = np.shares_memory(a, a);
            bool m = np.may_share_memory(a, a);
            Assert.IsTrue(s);
            Assert.IsTrue(m);
        }

        // ----------------------------------------------------------- candidate-cap DFS ----

        [TestMethod]
        public void CandidateCap_ThresholdMatchesNumPy()
        {
            // NumPy's own test (numpy/_core/tests/test_mem_overlap.py): two strided 3-D views of one
            // buffer genuinely overlap, but the exact solver needs >2 candidate solutions to prove it.
            // A small max_work budget therefore leaves the answer undecided.
            // >>> x = np.zeros([4,5,6], np.int8); a = x[:, ::2, ::3]; b = x[:, ::3, ::2]
            // shares/may default -> True; shares raises TooHardError at max_work 0..2, decides True at >=3.
            var x = np.zeros(new Shape(4, 5, 6)).astype(NPTypeCode.SByte);
            var a = x[":, ::2, ::3"];
            var b = x[":, ::3, ::2"];
            Assert.IsTrue(np.shares_memory(a, b));                 // default -1 (exact) decides True
            Assert.IsTrue(np.may_share_memory(a, b));              // default 0 (bounds) -> conservative True
            for (long mw = 0; mw <= 2; mw++)
                Assert.ThrowsException<TooHardError>(() => np.shares_memory(a, b, mw),
                    $"max_work={mw} should exhaust the candidate budget");
            for (long mw = 3; mw <= 8; mw++)
                Assert.IsTrue(np.shares_memory(a, b, mw), $"max_work={mw} should decide True");
            // may_share_memory never raises here — the undecided outcome folds into True.
            for (long mw = 0; mw <= 8; mw++)
                Assert.IsTrue(np.may_share_memory(a, b, mw));
        }

        // ------------------------------------------------------------- flag-state coverage ----

        [TestMethod]
        public void BroadcastViews_ShareByCollapsedExtent()
        {
            // A broadcast view's byte extent is only the SOURCE row it repeats (stride-0 axes add nothing),
            // so two broadcasts of the SAME row share, but broadcasts of DISJOINT rows do not even by bounds.
            var m = np.arange(24).reshape(4, 6);
            var row0a = np.broadcast_to(m["0:1, :"], new Shape(4, 6));
            var row0b = np.broadcast_to(m["0:1, :"], new Shape(4, 6));
            var row3 = np.broadcast_to(m["3:4, :"], new Shape(4, 6));
            Assert.IsFalse(row0a.flags.writeable);                 // broadcast views are read-only
            Assert.IsTrue(np.shares_memory(row0a, row0b));         // same source row -> share
            Assert.IsFalse(np.shares_memory(row0a, row3));         // rows 0 and 3 never touch
            Assert.IsFalse(np.may_share_memory(row0a, row3));      // disjoint extents -> bounds also False
        }

        [TestMethod]
        public void ReadOnlyView_SharesLikeWriteable()
        {
            // The WRITEABLE flag is irrelevant to memory sharing — the functions only read strides.
            var a = np.arange(12);
            var ro = a[":"];
            ro.flags.writeable = false;
            Assert.IsFalse(ro.flags.writeable);
            Assert.IsTrue(np.shares_memory(ro, a));
            Assert.IsFalse(np.shares_memory(ro, a.copy()));
        }

        [TestMethod]
        public void FContiguousLayouts_ShareOnlyWhenAView()
        {
            // A transpose is an F-contiguous VIEW (shares); asfortranarray of a C-contiguous 2-D array is a
            // fresh F-contiguous COPY (does not share) — the OWNDATA flag distinguishes them.
            var m = np.arange(12).reshape(3, 4);
            var t = m.T;
            Assert.IsTrue(t.flags.f_contiguous);
            Assert.IsFalse(t.flags.owndata);
            Assert.IsTrue(np.shares_memory(m, t));

            var f = np.asfortranarray(m);
            Assert.IsTrue(f.flags.f_contiguous);
            Assert.IsTrue(f.flags.owndata);
            Assert.IsFalse(np.shares_memory(m, f));
        }

        [TestMethod]
        public void MaxWork_LongMaxValue_IsValidHugeBudget()
        {
            // NumPy raises OverflowError for max_work=10**100 (it exceeds Py_ssize_t); that value is
            // unrepresentable in C#'s long, so the analogous extreme is long.MaxValue — a legitimate,
            // effectively-unlimited budget that decides exactly and never overflows.
            var a = np.arange(24);
            Assert.IsTrue(np.shares_memory(a, a["::2"], long.MaxValue));
            Assert.IsFalse(np.shares_memory(a["::2"], a["1::2"], long.MaxValue));
        }

        [TestMethod]
        [Misaligned]
        public void IntIndexZeroD_SharesInNumSharp_ButNumPyReturnsScalarCopy()
        {
            // DELIBERATE indexing-semantics divergence (NOT a shares_memory difference): a single-int index
            // returns a 0-d VIEW in NumSharp (shares storage) but a scalar COPY in NumPy (shares nothing).
            // The comparable 0-d VIEW built via reshape(()) shares in BOTH libraries.
            var a = np.arange(24);
            var zi = a["3"];                                       // NumSharp: 0-d view; NumPy a[3]: scalar copy
            Assert.AreEqual(0, zi.ndim);
            Assert.IsTrue(np.shares_memory(zi, a));                // NumSharp view shares; NumPy scalar would be False
            var zv = a["3:4"].reshape(new int[] { });             // 0-d view -> shares in NumPy too
            Assert.IsTrue(np.shares_memory(zv, a));
        }
    }
}
