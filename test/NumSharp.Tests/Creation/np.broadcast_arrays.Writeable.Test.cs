namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     NumPy 2.4.2 parity for <c>np.broadcast_arrays</c> WRITEABILITY.
    ///
    ///     <para>Unlike <c>np.broadcast_to</c> (which is ALWAYS read-only), NumPy 2.4.2's
    ///     <c>broadcast_arrays</c> returns arrays with <c>flags.writeable == True</c> for EVERY result —
    ///     even stretched (stride-0) ones — emitting only a <c>FutureWarning</c> that a future NumPy
    ///     version will change this. Probed directly against numpy 2.4.2:</para>
    ///     <code>
    ///     >>> x = np.array([1,2,3]); y = np.array([[10],[20]])
    ///     >>> a, b = np.broadcast_arrays(x, y)   # both (2,3), a.strides == (0,8)
    ///     >>> a.flags.writeable
    ///     True
    ///     >>> a[0,0] = 999
    ///     >>> a
    ///     array([[999,   2,   3],
    ///            [999,   2,   3]])              # stride-0 aliasing
    ///     >>> x
    ///     array([999,   2,   3])               # written through to the source
    ///     </code>
    ///     So the result stays broadcasted (stride-0 aliasing) yet writeable, and a write propagates
    ///     both across the broadcast axis AND back to the source array it views.
    /// </summary>
    [TestClass]
    public class BroadcastArraysWriteableTests : TestClass
    {
        // =====================================================================
        //  Writeability flag parity (writeable == True for ALL results)
        // =====================================================================

        [TestMethod]
        public void ParamsOverload_Stretched_IsWriteable_ButStaysBroadcasted()
        {
            // (3,) + (3,1) + (1,3) -> three (3,3) results, all stretched (stride-0).
            var xs = np.broadcast_arrays(new[] { np.arange(3), np.arange(3).reshape(3, 1), np.arange(3).reshape(1, 3) });
            xs.Length.Should().Be(3);
            foreach (var r in xs)
            {
                r.Shape.IsBroadcasted.Should().BeTrue("a stretched broadcast_arrays result still has a stride-0 axis");
                r.Shape.IsWriteable.Should().BeTrue("NumPy 2.4.2 broadcast_arrays results are writeable, even when stretched");
            }
        }

        [TestMethod]
        public void ParamsOverload_NoStretch_IsWriteable()
        {
            var ys = np.broadcast_arrays(new[] { np.arange(3), np.arange(3) });
            foreach (var r in ys)
                r.Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void TupleOverload_Stretched_IsWriteable()
        {
            var (l, r) = np.broadcast_arrays(np.arange(3), np.arange(3).reshape(3, 1));
            l.Shape.IsBroadcasted.Should().BeTrue();
            r.Shape.IsBroadcasted.Should().BeTrue();
            l.Shape.IsWriteable.Should().BeTrue();
            r.Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void TupleOverload_NoStretch_IsWriteable()
        {
            var (l, r) = np.broadcast_arrays(np.arange(3), np.arange(3));
            l.Shape.IsWriteable.Should().BeTrue();
            r.Shape.IsWriteable.Should().BeTrue();
        }

        // =====================================================================
        //  Write-through behavior parity (matches numpy 2.4.2 output byte-for-byte)
        // =====================================================================

        [TestMethod]
        public void Stretched_Write_AliasesAcrossBroadcastAxis_AndWritesToSource()
        {
            // x=(3,)[1,2,3], y=(2,1)[[10],[20]] -> a,b=(2,3); a.strides==(0,8)
            var x = np.array(new long[] { 1, 2, 3 });
            var y = np.array(new long[,] { { 10 }, { 20 } });
            var (a, b) = np.broadcast_arrays(x, y);

            a.Shape.IsWriteable.Should().BeTrue();

            a[0, 0] = 999L;

            // stride-0 aliasing: BOTH rows of column 0 read 999
            a.GetInt64(0, 0).Should().Be(999L);
            a.GetInt64(1, 0).Should().Be(999L);
            // untouched columns intact
            a.GetInt64(0, 1).Should().Be(2L);
            a.GetInt64(0, 2).Should().Be(3L);
            // written through to the source array (a views x)
            x.GetInt64(0).Should().Be(999L);
            x.GetInt64(1).Should().Be(2L);
            x.GetInt64(2).Should().Be(3L);
        }

        [TestMethod]
        public void NoStretch_Write_PropagatesToSource()
        {
            // Genuine view over a same-shape operand: writing propagates to the source.
            var c = np.array(new long[] { 1, 2, 3 });
            var d = np.array(new long[] { 4, 5, 6 });
            var (p, q) = np.broadcast_arrays(c, d);

            p.Shape.IsWriteable.Should().BeTrue();
            p[0] = 111L;
            c.GetInt64(0).Should().Be(111L);
        }
    }
}
