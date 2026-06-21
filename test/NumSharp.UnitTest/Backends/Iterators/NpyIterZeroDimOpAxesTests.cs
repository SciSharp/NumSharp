using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Coverage for the 0-dimensional / dropped-all-axes NpyIter configuration —
    /// i.e. <c>NpyIterRef.AdvancedNew(..., opAxesNDim: 0, opAxes: [[]])</c>, the shape produced
    /// by the IterAllButAxis pattern (NumPy's <c>NpyIter_RemoveAxis</c>) when the operand is 1-D
    /// and its only axis is dropped.
    ///
    /// NumPy semantics: removing every iterated axis leaves a 0-dimensional iterator with
    /// <c>NpyIter_GetIterSize() == 1</c> — exactly ONE iteration whose data pointer is the operand
    /// base (the dropped axis is the caller's responsibility, walked via its saved stride).
    ///
    /// BUG (pre-fix): CalculateBroadcastShape gated the op_axes branch on <c>opAxesNDim &gt; 0</c>,
    /// so <c>opAxesNDim == 0</c> fell through to natural broadcasting and returned the operand's
    /// own shape [N] → IterSize = N → the kernel was driven N times instead of once. For the sort
    /// driver (whose line kernel re-sorts the whole line per call) that made 1-D/axis=None sort and
    /// argsort O(N^2). These tests pin the correct 0-dim behavior at the iterator level.
    /// </summary>
    [TestClass]
    public class NpyIterZeroDimOpAxesTests
    {
        // ---------- probes carried through ForEach via auxdata (no captures) ----------
        private struct Probe { public long IterSize; public int NDim; public long Calls; public long TotalCount; public long FirstCount; }
        private struct Walk { public long Calls; public int N; public long ByteStride; public long Sum; }
        private struct Fill { public long Calls; public int N; public long ByteStride; public int Value; }
        private struct Copy { public long Calls; public int N; public long SrcStride; public long DstStride; }

        private static unsafe Probe RunProbe(NDArray[] ops, NpyIterPerOpFlags[] flags, int oaNdim, int[][] opAxes)
        {
            Probe pr = default;
            var iter = NpyIterRef.AdvancedNew(ops.Length, ops, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, flags, null, oaNdim, opAxes);
            try
            {
                pr.IterSize = iter.IterSize;
                pr.NDim = iter.NDim;
                NpyInnerLoopFunc k = (p, s, c, a) =>
                {
                    Probe* x = (Probe*)a;
                    x->Calls++; x->TotalCount += c;
                    if (x->Calls == 1) x->FirstCount = c;
                };
                iter.ForEach(k, &pr);
            }
            finally { iter.Dispose(); }
            return pr;
        }

        private static long ByteStride(NDArray a, int axis) => a.strides[axis] * a.dtypesize;

        // ============================ the bug: 1-D drop-only-axis ============================

        [TestMethod]
        public unsafe void OneD_DropOnlyAxis_IteratesExactlyOnce()
        {
            var a = np.arange(100).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READONLY }, 0, new[] { new int[0] });

            Assert.AreEqual(0, pr.NDim, "dropping the only axis => 0 iteration dimensions");
            Assert.AreEqual(1L, pr.IterSize, "0-dim iteration size must be 1 (bug produced 100)");
            Assert.AreEqual(1L, pr.Calls, "kernel must be called exactly once (bug called it N times => O(N^2))");
            Assert.AreEqual(1L, pr.FirstCount, "the single call's inner count is 1");
        }

        [TestMethod]
        public unsafe void OneD_DropOnlyAxis_LargeN_StillExactlyOneCall()
        {
            // The smoking gun for the O(N^2) blow-up: at N=100k the buggy path made 100k calls.
            var a = np.arange(100_000).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READONLY }, 0, new[] { new int[0] });

            Assert.AreEqual(1L, pr.IterSize);
            Assert.AreEqual(1L, pr.Calls, "100k-element 0-dim iterator must drive exactly one call, not 100k");
        }

        [TestMethod]
        public unsafe void OneD_DropOnlyAxis_SingleCall_BasePointerWalksWholeLine()
        {
            int N = 1000;
            var data = new int[N]; long expect = 0;
            for (int i = 0; i < N; i++) { data[i] = i * 3 - 7; expect += data[i]; }
            var a = np.array(data); // contiguous 1-D int32

            Walk w = default; w.N = N; w.ByteStride = ByteStride(a, 0);
            var iter = NpyIterRef.AdvancedNew(1, new[] { a }, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, new[] { NpyIterPerOpFlags.READONLY }, null, 0, new[] { new int[0] });
            try
            {
                NpyInnerLoopFunc k = (p, s, c, a2) =>
                {
                    Walk* x = (Walk*)a2; x->Calls++;
                    byte* line = (byte*)p[0]; long sum = 0;
                    for (int i = 0; i < x->N; i++) sum += *(int*)(line + i * x->ByteStride);
                    x->Sum = sum;
                };
                iter.ForEach(k, &w);
            }
            finally { iter.Dispose(); }

            Assert.AreEqual(1L, w.Calls, "exactly one call");
            Assert.AreEqual(expect, w.Sum, "p[0] must be the operand base so the whole line is reachable by stride");
        }

        [TestMethod]
        public unsafe void OneD_DropOnlyAxis_ReadWrite_SingleCall_WritesWholeLine()
        {
            int N = 500;
            var a = np.arange(N).astype(np.int32); // 0..499

            Fill f = default; f.N = N; f.ByteStride = ByteStride(a, 0); f.Value = 42;
            var iter = NpyIterRef.AdvancedNew(1, new[] { a }, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, new[] { NpyIterPerOpFlags.READWRITE }, null, 0, new[] { new int[0] });
            try
            {
                NpyInnerLoopFunc k = (p, s, c, a2) =>
                {
                    Fill* x = (Fill*)a2; x->Calls++;
                    byte* line = (byte*)p[0];
                    for (int i = 0; i < x->N; i++) *(int*)(line + i * x->ByteStride) = x->Value;
                };
                iter.ForEach(k, &f);
            }
            finally { iter.Dispose(); }

            Assert.AreEqual(1L, f.Calls, "exactly one call");
            for (int i = 0; i < N; i++)
                Assert.AreEqual(42, a.GetInt32(i), $"writeable base must let the single call fill element {i}");
        }

        [TestMethod]
        public unsafe void OneD_DropOnlyAxis_SlicedOffsetView_BaseIsOffsetCorrect()
        {
            var full = np.arange(20).astype(np.int32);
            var a = full["5:15"]; // elements 5..14, offset view (offset must be applied to the base ptr)
            long expect = 0; for (int i = 5; i < 15; i++) expect += i;

            Walk w = default; w.N = 10; w.ByteStride = ByteStride(a, 0);
            var iter = NpyIterRef.AdvancedNew(1, new[] { a }, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, new[] { NpyIterPerOpFlags.READONLY }, null, 0, new[] { new int[0] });
            try
            {
                NpyInnerLoopFunc k = (p, s, c, a2) =>
                {
                    Walk* x = (Walk*)a2; x->Calls++;
                    byte* line = (byte*)p[0]; long sum = 0;
                    for (int i = 0; i < x->N; i++) sum += *(int*)(line + i * x->ByteStride);
                    x->Sum = sum;
                };
                iter.ForEach(k, &w);
            }
            finally { iter.Dispose(); }

            Assert.AreEqual(1L, w.Calls);
            Assert.AreEqual(expect, w.Sum, "0-dim base pointer must include Shape.offset (expected sum of 5..14 = 95)");
        }

        [TestMethod]
        public unsafe void TwoOperand_DropOnlyAxis_IteratesOnce_BothBasesValid()
        {
            // The argsort shape: 2 lockstep operands (src readonly, dst writeonly), both 1-D, axis dropped.
            int N = 256;
            var src = np.arange(N).astype(np.int32);
            var dst = new NDArray(NPTypeCode.Int64, new Shape(N), false);

            Copy cp = default; cp.N = N; cp.SrcStride = ByteStride(src, 0); cp.DstStride = ByteStride(dst, 0);
            var iter = NpyIterRef.AdvancedNew(2, new[] { src, dst }, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }, null, 0, new[] { new int[0], new int[0] });
            try
            {
                NpyInnerLoopFunc k = (p, s, c, a2) =>
                {
                    Copy* x = (Copy*)a2; x->Calls++;
                    byte* sp = (byte*)p[0]; byte* dp = (byte*)p[1];
                    for (int i = 0; i < x->N; i++)
                        *(long*)(dp + i * x->DstStride) = *(int*)(sp + i * x->SrcStride);
                };
                iter.ForEach(k, &cp);
            }
            finally { iter.Dispose(); }

            Assert.AreEqual(1L, cp.Calls, "two-operand 0-dim iterator must drive exactly one lockstep call");
            for (int i = 0; i < N; i++)
                Assert.AreEqual((long)i, dst.GetInt64(i), $"both operand bases must be valid (dst[{i}])");
        }

        [TestMethod]
        public unsafe void TwoD_DropAllAxes_IteratesExactlyOnce()
        {
            // General 0-dim case (not just 1-D): drop BOTH axes of a 2-D operand.
            var a = np.arange(20).reshape(4, 5).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READONLY }, 0, new[] { new int[0] });

            Assert.AreEqual(0, pr.NDim);
            Assert.AreEqual(1L, pr.IterSize, "dropping all axes of a 20-element 2-D array => 1 iteration (bug gave 20)");
            Assert.AreEqual(1L, pr.Calls);
        }

        // ============================ regression guards (must hold before AND after the fix) ============================

        [TestMethod]
        public unsafe void Regression_NoOpAxes_1D_IteratesNTimes()
        {
            // opAxesNDim = -1 (unspecified): ordinary full iteration is unchanged by the fix.
            var a = np.arange(50).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READONLY }, -1, null);

            Assert.AreEqual(1, pr.NDim);
            Assert.AreEqual(50L, pr.IterSize);
            Assert.AreEqual(50L, pr.Calls, "normal 1-D iteration (no axis drop) still steps element-by-element");
        }

        [TestMethod]
        public unsafe void Regression_OpAxes_2D_DropAxis1_IteratesPerRow()
        {
            var a = np.arange(40).reshape(8, 5).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READWRITE }, 1, new[] { new[] { 0 } });

            Assert.AreEqual(1, pr.NDim);
            Assert.AreEqual(8L, pr.IterSize);
            Assert.AreEqual(8L, pr.Calls, "drop axis 1 of (8,5) => keep axis 0 => 8 line calls (already correct)");
        }

        [TestMethod]
        public unsafe void Regression_OpAxes_2D_DropAxis0_IteratesPerColumn()
        {
            var a = np.arange(40).reshape(8, 5).astype(np.int32);
            var pr = RunProbe(new[] { a }, new[] { NpyIterPerOpFlags.READWRITE }, 1, new[] { new[] { 1 } });

            Assert.AreEqual(1, pr.NDim);
            Assert.AreEqual(5L, pr.IterSize);
            Assert.AreEqual(5L, pr.Calls, "drop axis 0 of (8,5) => keep axis 1 => 5 line calls (already correct)");
        }
    }
}
