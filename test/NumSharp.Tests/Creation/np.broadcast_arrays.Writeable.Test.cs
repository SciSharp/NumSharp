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

        /// <summary>
        ///     A sub-array of a stretched result keeps its write access even where it is still broadcast (NumPy views
        ///     inherit WRITEABLE): with <c>a, b = np.broadcast_arrays(x[:, None], x)</c>, <c>a[0]</c> has stride <c>(0,)</c>,
        ///     reports <c>writeable=True</c>, and <c>a[0][...] = 9</c> writes x[0] (NumPy: <c>[9 2 3]</c>). It used to be
        ///     flagged read-only, so the store raised "assignment destination is read-only".
        /// </summary>
        [TestMethod]
        public void StretchedSubArray_InheritsWriteable()
        {
            var x = np.array(new long[] { 1, 2, 3 });
            var (a, b) = np.broadcast_arrays(x[Slice.All, Slice.NewAxis], x);
            var row = a[0];
            row.Shape.strides.Should().Equal(new long[] { 0 });
            row.Shape.IsBroadcasted.Should().BeTrue();
            row.Shape.IsWriteable.Should().BeTrue("a sub-array of a writeable broadcast result inherits its write access");
            row[Slice.Ellipsis] = 9L;
            x.GetInt64(0).Should().Be(9L);
            x.GetInt64(1).Should().Be(2L);
            x.GetInt64(2).Should().Be(3L);
        }

        /// <summary>
        ///     A whole-view store into a stretched sub-array that starts at a NON-zero offset lands on that sub-array's own
        ///     element (found 2026-10-13 by the string oracle's storenone section). With
        ///     <c>a, b = np.broadcast_arrays(x[:, None], x)</c> and <c>x = np.arange(3)</c>, NumPy 2.4.2 gives
        ///     <c>a[1][...] = 9</c> → <c>[0 9 2]</c>, <c>a[2][...] = [7, 8, 9]</c> → <c>[0 1 9]</c> (the stride-0 row keeps the
        ///     last value), and <c>b[r][...] = 9</c> → <c>[9 9 9]</c> for every row. The sub-array view of an empty index list
        ///     (the whole-view store's target) used to drop the row's offset, so every store wrote <c>x[0]</c>; only row 0, the
        ///     case the test above covers, was right.
        /// </summary>
        [TestMethod]
        public void StretchedSubArray_AtAnOffset_StoresIntoItsOwnElement()
        {
            foreach (var which in new[] { "a", "b" })
            foreach (var row in new[] { 0, 1, 2 })
            foreach (var form in new[] { "scalar", "array", "SetData" })
            {
                var x = np.arange(3L);
                var (a, b) = np.broadcast_arrays(x[Slice.All, Slice.NewAxis], x);
                var target = (which == "a" ? a : b)[row];
                // The same empty-index sub-array, read through the coordinate shim: a's row reads x[row] three times.
                using (var whole = target.GetData(new int[0]))
                    whole.ToArray<long>().Should().Equal(which == "a" ? new long[] { row, row, row } : new long[] { 0, 1, 2 },
                        $"{which}[{row}].GetData(()) reads its own elements");
                switch (form)
                {
                    case "scalar": target[Slice.Ellipsis] = NDArray.Scalar(9L); break;
                    case "array": target[Slice.Ellipsis] = np.array(new long[] { 7, 8, 9 }); break;
                    default: target.SetData(NDArray.Scalar(9L)); break;
                }

                long[] expected;
                if (which == "a")
                {
                    // a's rows are stride-0 views of x[row]: every slot aliases it, the last value stays.
                    expected = new long[] { 0, 1, 2 };
                    expected[row] = 9;
                }
                else
                    expected = form == "array" ? new long[] { 7, 8, 9 } : new long[] { 9, 9, 9 };
                x.ToArray<long>().Should().Equal(expected, $"{which}[{row}][...] = ({form})");
            }
        }
    }
}
