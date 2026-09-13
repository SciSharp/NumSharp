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
    }
}
