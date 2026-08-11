using System;
using System.IO;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     Every write path that reaches an array's raw memory through an IL/SIMD kernel (bypassing
    ///     the guarded per-element setters) must reject a NON-WRITEABLE target, exactly as NumPy does.
    ///
    ///     <para>This pins a MEMORY-SAFETY bug, not a cosmetic one. A non-writeable array in NumSharp
    ///     is produced by <c>np.broadcast_to</c>, by <c>np.load(mmap_mode:"r")</c>, and by the pythonnet
    ///     interop over read-only Python buffers. Before these guards, an in-place ufunc / <c>put</c> /
    ///     <c>place</c> / <c>shuffle</c> / <c>evaluate(out=)</c> wrote straight through such a target:
    ///     it silently corrupted the shared source (e.g. an immutable Python <c>bytes</c>) and, on a
    ///     read-only <c>mmap('r')</c> whose pages are <c>PROT_READ</c>, took the whole process down with
    ///     an access violation. NumPy raises <c>ValueError: output array is read-only</c> instead; the
    ///     messages here are verbatim NumPy 2.4.2 (NumSharp maps the ValueError onto
    ///     <see cref="NumSharpException"/>, the house convention for read-only writes).</para>
    /// </summary>
    [TestClass]
    public class ReadonlyWriteGuardTests
    {
        /// <summary>A read-only view over <paramref name="a"/> — NumPy's broadcast_to is ALWAYS read-only.</summary>
        private static NDArray Ro(NDArray a) => np.broadcast_to(a, a.Shape);

        private static void ShouldThrowReadonly(Action act, string message)
            => act.Should().Throw<NumSharpException>().WithMessage(message);

        // =====================================================================
        //  np.broadcast_to writeability — NumPy parity: ALWAYS read-only
        // =====================================================================

        [TestMethod]
        public void BroadcastTo_SameShape_IsReadonly()
        {
            // NumPy: np.broadcast_to(x, x.shape).flags.writeable == False, even with no stretching.
            var a = np.arange(3);
            np.broadcast_to(a, a.Shape).Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void BroadcastTo_Stretch_IsReadonly()
        {
            var a = np.arange(3);
            np.broadcast_to(a, new Shape(2, 3)).Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void BroadcastTo_2dRow_IsReadonly()
        {
            var a = np.arange(3).reshape(1, 3);
            np.broadcast_to(a, new Shape(4, 3)).Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void OwnedArray_IsWriteable()
        {
            np.arange(3).Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void BroadcastTo_SameShape_DoesNotCorruptSourceThroughUfunc()
        {
            var src = np.arange(3).astype(NPTypeCode.Double);      // owns its data
            var view = np.broadcast_to(src, src.Shape);            // read-only view sharing src's memory
            try { np.add(view, view, view); } catch (NumSharpException) { }
            src.GetDouble(0).Should().Be(0.0);
            src.GetDouble(1).Should().Be(1.0);
            src.GetDouble(2).Should().Be(2.0);
        }

        // =====================================================================
        //  Elementwise ufunc out= (binary / unary / comparison / bitwise / where)
        // =====================================================================

        [TestMethod]
        public void Ufunc_Add_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.add(a, a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Multiply_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.multiply(a, np.array(2.0), ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Negative_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.negative(a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Sqrt_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.sqrt(a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Comparison_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Boolean));
            ShouldThrowReadonly(() => np.greater(a, a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_BitwiseAnd_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Int32);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Int32));
            ShouldThrowReadonly(() => np.bitwise_and(a, a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Invert_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Int32);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Int32));
            ShouldThrowReadonly(() => np.invert(a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Positive_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.positive(a, ro), "output array is read-only");
        }

        [TestMethod]
        public void Ufunc_Add_withWhere_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            var mask = np.ones(new Shape(3)).astype(NPTypeCode.Boolean);
            ShouldThrowReadonly(() => np.add(a, a, ro, mask), "output array is read-only");
        }

        [TestMethod]
        public void Round_out_readonly_throws()
        {
            var a = (np.arange(3).astype(NPTypeCode.Double) + 0.4);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.round_(a, 0, ro), "output array is read-only");
        }

        [TestMethod]
        public void Around_out_readonly_throws()
        {
            var a = (np.arange(3).astype(NPTypeCode.Double) + 0.4);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.around(a, 0, ro), "output array is read-only");
        }

        // =====================================================================
        //  np.evaluate(out=)
        // =====================================================================

        [TestMethod]
        public void Evaluate_out_readonly_throws()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var ro = Ro(np.zeros(new Shape(3)).astype(NPTypeCode.Double));
            ShouldThrowReadonly(() => np.evaluate((NDExpr)a + a, @out: ro), "output array is read-only");
        }

        // =====================================================================
        //  In-place scatter / shuffle
        // =====================================================================

        [TestMethod]
        public void Put_readonly_throws()
        {
            var ro = Ro(np.arange(3));
            ShouldThrowReadonly(() => np.put(ro, np.array(new long[] { 0 }), np.array(new int[] { 9 })),
                "put: output array is read-only");
        }

        [TestMethod]
        public void Put_readonly_emptyIndices_stillThrows()
        {
            // NumPy checks writeability BEFORE the empty-indices no-op.
            var ro = Ro(np.arange(3));
            ShouldThrowReadonly(() => np.put(ro, np.array(new long[] { }), np.array(new int[] { })),
                "put: output array is read-only");
        }

        [TestMethod]
        public void Place_readonly_throws()
        {
            var ro = Ro(np.arange(3));
            ShouldThrowReadonly(() => np.place(ro, np.ones(new Shape(3)).astype(NPTypeCode.Boolean), np.array(new int[] { 9 })),
                "WRITEBACKIFCOPY base is read-only");
        }

        [TestMethod]
        public void Shuffle_readonly_throws()
        {
            var ro = Ro(np.arange(8));
            ShouldThrowReadonly(() => np.random.shuffle(ro), "array is read-only");
        }

        [TestMethod]
        public void Put_readonly_doesNotCorruptSource()
        {
            var src = np.arange(3).astype(NPTypeCode.Int32);
            var ro = np.broadcast_to(src, src.Shape);
            try { np.put(ro, np.array(new long[] { 0 }), np.array(new int[] { 99 })); } catch (NumSharpException) { }
            src.GetInt32(0).Should().Be(0);
        }

        // =====================================================================
        //  Regression: WRITEABLE targets must still compute
        // =====================================================================

        [TestMethod]
        public void Ufunc_out_writable_stillWorks()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var outArr = np.zeros(new Shape(3)).astype(NPTypeCode.Double);
            var r = np.add(a, a, outArr);
            ReferenceEquals(r, outArr).Should().BeTrue();
            outArr.GetDouble(2).Should().Be(4.0);
        }

        [TestMethod]
        public void Put_writable_stillWorks()
        {
            var a = np.arange(3).astype(NPTypeCode.Int32);
            np.put(a, np.array(new long[] { 0 }), np.array(new int[] { 9 }));
            a.GetInt32(0).Should().Be(9);
        }

        [TestMethod]
        public void Place_writable_stillWorks()
        {
            var a = np.arange(3).astype(NPTypeCode.Int32);
            np.place(a, np.array(new bool[] { true, false, false }), np.array(new int[] { 55 }));
            a.GetInt32(0).Should().Be(55);
        }

        [TestMethod]
        public void Shuffle_writable_stillWorks()
        {
            var a = np.arange(8).astype(NPTypeCode.Int32);
            np.random.shuffle(a);
            long sum = 0;
            for (int i = 0; i < 8; i++) sum += a.GetInt32(i);
            sum.Should().Be(28);   // a permutation of 0..7
        }

        [TestMethod]
        public void Evaluate_out_writable_stillWorks()
        {
            var a = np.arange(3).astype(NPTypeCode.Double);
            var outArr = np.zeros(new Shape(3)).astype(NPTypeCode.Double);
            np.evaluate((NDExpr)a + a, @out: outArr);
            outArr.GetDouble(2).Should().Be(4.0);
        }

        // =====================================================================
        //  Real-world: a read-only memmap (PROT_READ) must THROW, never crash
        // =====================================================================

        [TestMethod]
        public void MmapReadonly_ufunc_out_throws_notCrash()
        {
            string path = Path.Combine(Path.GetTempPath(), "ns_ro_" + Guid.NewGuid().ToString("N") + ".npy");
            try
            {
                np.save(path, np.arange(4).astype(NPTypeCode.Double));
                var ro = (NDArray)np.load(path, mmap_mode: "r");
                ro.Shape.IsWriteable.Should().BeFalse();
                ShouldThrowReadonly(() => np.negative(ro, ro), "output array is read-only");
            }
            finally
            {
                try { File.Delete(path); } catch { /* best effort */ }
            }
        }

        [TestMethod]
        public void MmapReadonly_put_throws_notCrash()
        {
            string path = Path.Combine(Path.GetTempPath(), "ns_ro_" + Guid.NewGuid().ToString("N") + ".npy");
            try
            {
                np.save(path, np.arange(4).astype(NPTypeCode.Int32));
                var ro = (NDArray)np.load(path, mmap_mode: "r");
                ShouldThrowReadonly(() => np.put(ro, np.array(new long[] { 0 }), np.array(new int[] { 9 })),
                    "put: output array is read-only");
            }
            finally
            {
                try { File.Delete(path); } catch { /* best effort */ }
            }
        }
    }
}
