using System;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     Pins what a FAILED iteration leaves in its operands: NumPy's <c>NpyIter_Deallocate</c> writes the held buffers back
    ///     and resolves WRITEBACKIFCOPY only when no error is set; when one is set it clears the buffers
    ///     (<c>npyiter_clear_buffers</c>) and discards the write-backs, so an operand keeps what the completed fills wrote and
    ///     nothing of the failed one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy 2.4.2, probed with object loops (its loops that can raise in the middle of a call):
    ///     <c>np.add(objects_with_a_raising_element, 1.0, out=np.full(4, -7.0), casting='unsafe')</c> raises and leaves the
    ///     float64 out <c>[-7, -7, -7, -7]</c>; with <c>3 * np.getbufsize()</c> elements and the raising one at <c>n - 5</c>, the
    ///     two completed fills (16,384 of 24,576 elements) reach the out and the failed third does not; an out that overlaps
    ///     its input (<c>np.add(a[:-1], 1.0, out=a[1:])</c>, a WRITEBACKIFCOPY temporary) leaves <c>a</c> unchanged; and a
    ///     <c>with np.nditer(...)</c> block whose body raises still writes its buffered operand back.
    ///     </para>
    ///     <para>
    ///     NumSharp's analog of a raising loop is the decimal division, which throws <see cref="DivideByZeroException"/> at a
    ///     zero divisor. Before the fix <c>NDIterRef.Dispose</c> wrote the failed fill's buffer back: the out received the
    ///     half-computed decimal buffer, and when an element the kernel never wrote held a garbage decimal (a scale byte past
    ///     28) the write-back's <c>Decimal.VarR8FromDec</c> raised <see cref="IndexOutOfRangeException"/> in place of the
    ///     <see cref="DivideByZeroException"/> (15 of 4,000 calls in a probe); the overlapping out was overwritten by an
    ///     uninitialized temporary.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class NDIterFailedIterationTests
    {
        /// <summary>A decimal array of the given values.</summary>
        /// <param name="values">The values (exact in decimal).</param>
        /// <returns>A new 1-D decimal array.</returns>
        private static NDArray Decimals(params double[] values) => np.array(values.Select(v => (decimal)v).ToArray());

        /// <summary>
        ///     A buffered out (a float64 out of the decimal loop) of a call whose kernel throws at its third element: every
        ///     call raises <see cref="DivideByZeroException"/>, and the out keeps its values (NumPy's object-loop probe:
        ///     <c>[-7, -7, -7, -7]</c>). Repeated, because the garbage that made the old write-back raise another exception
        ///     depends on what the pooled buffer held before.
        /// </summary>
        [TestMethod]
        public void TrueDivide_DecimalZeroDivisor_Float64Out_RaisesAndLeavesOutUnchanged()
        {
            for (int i = 0; i < 200; i++)
            {
                using var x1 = Decimals(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).reshape(3, 4);
                using var x2 = Decimals(1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1).reshape(3, 4);
                using var f = np.asfortranarray(x1);   // the layout the snapshot found the defect on
                using var o = np.full(new Shape(3, 4), -7.0, np.float64);

                Assert.ThrowsExactly<DivideByZeroException>(() => np.true_divide(f, x2, o));
                o.ToArray<double>().Should().AllBeEquivalentTo(-7.0, $"call {i}: a failed fill is never written back");
            }
        }

        /// <summary>
        ///     The failing element in the THIRD fill of the buffered out: the two fills the iteration completed were written
        ///     back as it moved past them, the third is dropped (NumPy: 16,384 of 24,576 elements written).
        /// </summary>
        [TestMethod]
        public void TrueDivide_DecimalZeroDivisor_InTheThirdFill_KeepsTheCompletedFills()
        {
            long buf = np.getbufsize();
            int n = checked((int)(3 * buf));
            var num = new decimal[n];
            var den = new decimal[n];
            for (int i = 0; i < n; i++)
            {
                num[i] = i + 1;
                den[i] = 1;
            }
            den[n - 5] = 0;
            using var x1 = np.array(num);
            using var x2 = np.array(den);
            using var o = np.full(new Shape(n), -7.0, np.float64);

            Assert.ThrowsExactly<DivideByZeroException>(() => np.true_divide(x1, x2, o));

            var got = o.ToArray<double>();
            for (int i = 0; i < 2 * buf; i++)
                got[i].Should().Be(i + 1, "element {0} belongs to a fill the iteration completed", i);
            for (long i = 2 * buf; i < n; i++)
                got[i].Should().Be(-7.0, "element {0} belongs to the failed fill", i);
        }

        /// <summary>
        ///     An out that overlaps its input (COPY_IF_OVERLAP gives the out a write-back temporary): the failed call's
        ///     temporary is discarded and the array keeps its values (NumPy: <c>a</c> unchanged).
        /// </summary>
        [TestMethod]
        public void TrueDivide_DecimalZeroDivisor_OverlappingOut_LeavesTheArrayUnchanged()
        {
            using var a = Decimals(1, 2, 3, 4, 5);
            using var b = Decimals(1, 1, 0, 1);

            Assert.ThrowsExactly<DivideByZeroException>(() => np.true_divide(a["0:4"], b, a["1:5"]));

            a.ToArray<decimal>().Should().Equal(1m, 2m, 3m, 4m, 5m);
        }

        /// <summary>
        ///     The flag's boundary: an exception in the body of a loop the caller drives (np.nditer's element walk) is not a
        ///     failed iteration, and the buffered operand is written back when the iterator is disposed, as NumPy's
        ///     <c>with np.nditer(...)</c> block writes it back when its body raises (probed: the first element written, then
        ///     a raise, gives <c>[100, 1, 2, 3, 4, 5]</c>). The body raises at the first element, so the test does not depend
        ///     on which element np.nditer's protocol publishes after k steps (the surface tier's gap SU42).
        /// </summary>
        [TestMethod]
        public void NDIter_ExceptionInTheCallersLoop_StillWritesTheBufferBack()
        {
            using var b = np.arange(6.0);
            try
            {
                using var it = np.nditer(b, new[] { "buffered" }, new[] { "readwrite" }, new DType[] { np.float32 },
                    casting: "same_kind");
                // Each step yields one view per operand (NumPy's tuple for several operands), here the one operand's.
                foreach (NDArray[] step in it)
                {
                    step[0][Slice.Ellipsis] = 100.0f;
                    throw new InvalidOperationException("stop");
                }
            }
            catch (InvalidOperationException)
            {
            }

            b.ToArray<double>().Should().Equal(100.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }
    }
}
