using System;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    /// argsort (and <c>np.ma.sort</c>, which is argsort + take_along_axis) of an array with NO lines:
    /// the sort axis has ≥ 2 elements but some OTHER dimension is 0, e.g. <c>(0, 3)</c> along the last
    /// axis. Expected shapes/dtypes probed against NumPy 2.4.2: the result keeps the input's dims
    /// (or <c>(0,)</c> for <c>axis=None</c>) and is int64, for every dtype.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Regression gate for a native heap overrun. <c>ArgSortInto</c> guarded only an empty AXIS
    /// (<c>N == 0</c>), not an empty ARRAY. With zero lines the all-but-axis iterator still made
    /// one kernel call, and the radix line kernels ignore the per-call count, so each call wrote
    /// a full line of N int64 indices into the result's ZERO-byte buffer. glibc never noticed;
    /// Windows' heap manager killed the process with <c>0xC0000374</c> (STATUS_HEAP_CORRUPTION) on
    /// a later allocation, which crashed the Oracle suite's test host in whatever test ran next
    /// (bisected to the <c>ma.sort/empty_2d/bool</c> corpus case).
    /// </para>
    /// <para>
    /// The corruption is not observable from managed code, so a regression shows up the same
    /// way the bug did: the test host dies on Windows. <see cref="ChurnNativeHeap"/> makes the
    /// heap manager revalidate blocks right after each call, so the crash surfaces inside these
    /// tests rather than in an unrelated one later.
    /// </para>
    /// </remarks>
    [TestClass]
    public class ArgsortZeroLinesTests
    {
        /// <summary>One dtype per argsort kernel family: 4-byte radix, 8-byte radix, the float
        /// NaN-last radix (single and double), and the managed comparison sort (Half/Complex/Decimal).</summary>
        private static readonly NPTypeCode[] KernelFamilies =
        {
            NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Char, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
            NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal,
        };

        /// <summary>
        /// Allocate and free a burst of small native buffers so Windows' heap manager validates the
        /// blocks next to anything a preceding call overran. Costs about a millisecond.
        /// </summary>
        private static void ChurnNativeHeap()
        {
            for (int k = 0; k < 400; k++)
            {
                using var t = np.zeros(new Shape(k % 61 + 1), NPTypeCode.Double);
            }
        }

        /// <summary>
        /// Argsort every zero-line shape along its lines for every kernel family. Each call must
        /// return an empty int64 array with the input's dims and must not write into it.
        /// </summary>
        [TestMethod]
        public void Argsort_ZeroLines_ReturnsEmptyInt64_WithoutWritingALine()
        {
            // (dims, axis): (0,3)/-1 and (2,0,3)/-1 are the overrun shapes (a length-3 line, zero
            // lines); the others are empty-array neighbours that must keep working.
            var cases = new (long[] dims, int axis)[]
            {
                (new long[] { 0, 3 }, -1), (new long[] { 2, 0, 3 }, -1), (new long[] { 2, 0, 3 }, 1),
                (new long[] { 0, 3 }, 0), (new long[] { 3, 0 }, -1), (new long[] { 0, 1 }, -1),
            };

            foreach (var tc in KernelFamilies)
            foreach (var (dims, axis) in cases)
            {
                for (int rep = 0; rep < 20; rep++)
                {
                    using var a = np.zeros(new Shape(dims), tc);
                    using var r = np.argsort(a, axis);
                    Assert.AreEqual(NPTypeCode.Int64, r.typecode, $"{tc} {string.Join("x", dims)} axis={axis}: dtype");
                    CollectionAssert.AreEqual(dims, r.shape, $"{tc} {string.Join("x", dims)} axis={axis}: shape");
                    Assert.AreEqual(0L, r.size, $"{tc} {string.Join("x", dims)} axis={axis}: size");
                    ChurnNativeHeap();
                }
            }
        }

        /// <summary>
        /// <c>axis=None</c> flattens first, so a zero-line input is a plain empty 1-D argsort —
        /// NumPy returns <c>(0,)</c> int64.
        /// </summary>
        [TestMethod]
        public void Argsort_ZeroLines_AxisNone_IsEmptyVector()
        {
            foreach (var tc in KernelFamilies)
            {
                using var a = np.zeros(new Shape(0, 3), tc);
                using var r = np.argsort(a, null);
                Assert.AreEqual(NPTypeCode.Int64, r.typecode, $"{tc}: dtype");
                CollectionAssert.AreEqual(new long[] { 0 }, r.shape, $"{tc}: shape");
                ChurnNativeHeap();
            }
        }

        /// <summary>
        /// The corpus case that exposed the overrun: <c>np.ma.sort</c> of an empty <c>(0, 3)</c> bool
        /// masked array (<c>endwith=True</c>) — argsort of the filled keys, then take_along_axis of
        /// data and mask. NumPy 2.4.2 returns an empty <c>(0, 3)</c> bool masked array.
        /// </summary>
        [TestMethod]
        public void MaSort_EmptyTwoDimensional_ReturnsEmptyMaskedArray()
        {
            for (int rep = 0; rep < 20; rep++)
            {
                using var data = np.zeros(new Shape(0, 3), NPTypeCode.Boolean);
                using var mask = np.zeros(new Shape(0, 3), NPTypeCode.Boolean);
                var sorted = np.ma.sort(np.ma.masked_array(data, mask), -1, true);
                CollectionAssert.AreEqual(new long[] { 0, 3 }, sorted.shape);
                Assert.AreEqual(NPTypeCode.Boolean, sorted.typecode);
                ChurnNativeHeap();
            }
        }
    }
}
