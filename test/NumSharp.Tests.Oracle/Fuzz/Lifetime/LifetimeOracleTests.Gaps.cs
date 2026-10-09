using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The OPEN lifetime bugs, one [OpenBugs] test per documented gap (<c>GAPS</c> in
    ///     <c>test/oracle/gen_lifetime_oracle.py</c>): each replays its carved cases from <c>lifetime_gaps.jsonl</c> in the
    ///     lifetime child and fails while any of them still diverges, crashes or throws.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>What each gap is.</b> A NumSharp method reads an operand's raw buffer pointer and runs its kernel over it,
    ///     and neither it nor its callers use the operand afterwards. Optimized code then stops reporting the operand to
    ///     the GC at the pointer read; a collection during the kernel (a second thread allocating, or the pool's own GC
    ///     pacing) finds the operand unreachable when the caller passed a temporary, and the finalizer of its memory block
    ///     returns the buffer to the pool or to the OS while the kernel still uses it: a silently wrong result, a spurious
    ///     IndexError from poisoned indices, or a fatal access violation. Every operand named below is one the caller can
    ///     pass as a temporary (the method uses it as-is, without a copy it owns).
    ///     </para>
    ///     <para>
    ///     <b>The fix pattern</b> is a <c>GC.KeepAlive(operand)</c> after the kernel call in the method that reads the
    ///     pointer (np.nonzero, np.argwhere and np.unravel_index already have it). When a gap is fixed its test passes:
    ///     drop the gap's rule from <c>_gap_of</c>, regenerate, and its cases join the gate (<see cref="Lifetime"/>).
    ///     </para>
    ///     <para>
    ///     These tests are excluded from CI like every [OpenBugs] test. Each needs the child process (tiered compilation
    ///     off) and a few seconds. Detection is a race the stress thread wins most of the time: a run in which a gap's
    ///     cases all happen to survive passes, so confirm a "fixed" gap with a second run before retiring its rule.
    ///     </para>
    /// </remarks>
    public partial class LifetimeOracleTests
    {
        /// <summary>
        ///     LT01 — <c>np.take</c>: <c>ExecuteTakeKernel</c> reads the source and the index buffer (int64, or int32 read in
        ///     place) and gathers through them; a temporary source or index array can be freed mid-gather (every mode, every
        ///     dtype, strings included).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT01 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT01_Take() => AssertGapsClosed("LT01");

        /// <summary>LT02 — <c>np.put</c>: <c>PutKernel</c> reads the index and value buffers it scatters from.</summary>
        /// <exception cref="AssertFailedException">A carved LT02 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT02_Put() => AssertGapsClosed("LT02");

        /// <summary>LT03 — <c>np.place</c>: <c>ExecutePlace</c> reads the caller's bool mask and values as-is.</summary>
        /// <exception cref="AssertFailedException">A carved LT03 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT03_Place() => AssertGapsClosed("LT03");

        /// <summary>LT04 — <c>np.putmask</c>: <c>ExecutePutMask</c> reads the caller's bool mask and values as-is.</summary>
        /// <exception cref="AssertFailedException">A carved LT04 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT04_Putmask() => AssertGapsClosed("LT04");

        /// <summary>LT05 — <c>np.take_along_axis</c>: the gather reads the source and the index buffer.</summary>
        /// <exception cref="AssertFailedException">A carved LT05 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT05_TakeAlongAxis() => AssertGapsClosed("LT05");

        /// <summary>
        ///     LT06 — <c>np.put_along_axis</c>: <c>PutAlongAxisCore</c> reads the index and value buffers (a freed index
        ///     buffer makes the scatter write out of bounds: a fatal access violation).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT06 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT06_PutAlongAxis() => AssertGapsClosed("LT06");

        /// <summary>
        ///     LT07 — <c>np.where(condition, x, y)</c> (numeric dtypes): <c>WhereKernelDispatch</c> reads cond, x and y. The
        ///     scalar-x / scalar-y dispatches are kept alive by their own fallback branch; the string route uses NDIter.
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT07 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT07_Where() => AssertGapsClosed("LT07");

        /// <summary>LT08 — <c>np.copyto(dst, src, where=mask)</c>: <c>CopytoMaskedRaw</c> reads the source and the mask.</summary>
        /// <exception cref="AssertFailedException">A carved LT08 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT08_CopytoWhere() => AssertGapsClosed("LT08");

        /// <summary>
        ///     LT09 — <c>np.copyto</c> across dtypes: <c>NDIter.Copy</c>'s contiguous cast kernel reads the source storage
        ///     with no later use (a shared path: assignments that cast run through it too). A short window — a rare hit.
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT09 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT09_CopytoCast() => AssertGapsClosed("LT09");

        /// <summary>LT10 — <c>np.max(a, initial=, where=)</c>: <c>RunMaxMin</c> reads the input buffer.</summary>
        /// <exception cref="AssertFailedException">A carved LT10 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT10_MaxInitialWhere() => AssertGapsClosed("LT10");

        /// <summary>LT11 — <c>np.max</c> of float16: <c>HalfMinMaxReduce</c> reads the input buffer.</summary>
        /// <exception cref="AssertFailedException">A carved LT11 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT11_MaxFloat16() => AssertGapsClosed("LT11");

        /// <summary>
        ///     LT12 — <c>np.cumsum</c> / <c>np.cumprod</c> without an axis: <c>ScanElementwiseFlat</c> reads the input (the
        ///     axis route is kept alive).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT12 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT12_CumsumCumprod() => AssertGapsClosed("LT12");

        /// <summary>LT13 — <c>np.nancumsum</c>: <c>NanReplace.ReplaceContiguousFloat</c> reads the input.</summary>
        /// <exception cref="AssertFailedException">A carved LT13 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT13_Nancumsum() => AssertGapsClosed("LT13");

        /// <summary>LT14 — <c>np.searchsorted</c>: the search kernel reads the sorted array and the keys.</summary>
        /// <exception cref="AssertFailedException">A carved LT14 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT14_Searchsorted() => AssertGapsClosed("LT14");

        /// <summary>LT15 — <c>np.digitize</c>: the bins' monotonicity scan and the search read x and the bins.</summary>
        /// <exception cref="AssertFailedException">A carved LT15 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT15_Digitize() => AssertGapsClosed("LT15");

        /// <summary>
        ///     LT16 — <c>np.bincount</c>: <c>MinMaxInt64</c>, <c>BincountCount</c> and <c>BincountWeighted</c> read x and the
        ///     weights (a freed x makes the count write out of bounds: a fatal access violation).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT16 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT16_Bincount() => AssertGapsClosed("LT16");

        /// <summary>LT17 — <c>np.reciprocal</c> of integers: <c>ReciprocalInteger</c> reads the input (floats are kept alive).</summary>
        /// <exception cref="AssertFailedException">A carved LT17 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT17_ReciprocalInteger() => AssertGapsClosed("LT17");

        /// <summary>LT18 — <c>np.dot</c> / <c>np.inner</c> of two vectors: <c>DotInner1D</c> reads both.</summary>
        /// <exception cref="AssertFailedException">A carved LT18 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT18_DotInner1D() => AssertGapsClosed("LT18");

        /// <summary>
        ///     LT19 — <c>np.dot(matrix, vector)</c>, <c>np.matvec</c>, <c>np.vecmat</c>: the managed product reads the MATRIX
        ///     operand without keeping it alive (the vector operand is kept alive).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT19 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT19_MatrixVector() => AssertGapsClosed("LT19");

        /// <summary>LT20 — <c>np.dot(matrix, matrix)</c>: <c>DotNDMDSimdDouble</c> reads both operands.</summary>
        /// <exception cref="AssertFailedException">A carved LT20 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT20_DotMatrixMatrix() => AssertGapsClosed("LT20");

        /// <summary>
        ///     LT21 — <c>np.linalg.det</c> / <c>inv</c> / <c>solve</c> without a backend: <c>ManagedLu</c> reads a (and b);
        ///     a poisoned matrix also surfaces as a spurious <c>LinAlgError: Singular matrix</c>.
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT21 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT21_ManagedLu() => AssertGapsClosed("LT21");

        /// <summary>
        ///     LT22 — <c>np.convolve</c> / <c>np.correlate</c>: <c>SlidingCorrelate</c> reads the data and the kernel
        ///     (convolve's kernel is reversed into a copy it owns, so only its data operand is lost).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT22 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT22_Sliding() => AssertGapsClosed("LT22");

        /// <summary>
        ///     LT23 — <c>np.polynomial.polynomial.polyder</c>: <c>NDPolyCalc.TryDirectSource</c> reads the caller's series in
        ///     place for the load stage (polyint takes that route only for a fused scale; its default route copies first).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT23 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT23_PolyCalculus() => AssertGapsClosed("LT23");

        /// <summary>LT24 — <c>np.polynomial.polynomial.polyvander</c>: <c>NDPolyVander</c> reads x in place.</summary>
        /// <exception cref="AssertFailedException">A carved LT24 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT24_Polyvander() => AssertGapsClosed("LT24");

        /// <summary>
        ///     LT25 — the <c>!</c> operator of a C-contiguous bool array: <c>NDArray.operator !</c> reads its operand in a
        ///     raw loop (other layouts route through np.logical_not, which is kept alive).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT25 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT25_NotOperator() => AssertGapsClosed("LT25");

        /// <summary>
        ///     LT26 — <c>np.all</c> of a bool array without an axis: <c>NDIter.ReduceBool</c> scans the input buffer with no
        ///     later use of it (a short window: the scan is fast — a rare hit).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT26 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT26_AllFlat() => AssertGapsClosed("LT26");

        /// <summary>
        ///     LT27 — <c>NDArray.ToMuliDimArray&lt;T&gt;()</c>: the copy reads <c>Storage.Address</c> of the receiver and never
        ///     touches the receiver again, so a temporary receiver can be freed mid-copy (a short window — a rare hit).
        /// </summary>
        /// <exception cref="AssertFailedException">A carved LT27 case still diverges (the gap is open).</exception>
        [TestMethod, TestCategory("OpenBugs")]
        public void Lifetime_LT27_ToMuliDimArray() => AssertGapsClosed("LT27");
    }
}
