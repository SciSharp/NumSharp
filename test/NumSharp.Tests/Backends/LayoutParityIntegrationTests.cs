using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     Cross-API integration coverage for the modelled numpy-internal representations
    ///     (journey3 53b5d82e): the LayoutParityOracleTests corpus proves each surface matches
    ///     NumPy bit-for-bit in isolation; THIS class proves the new semantics compose with the
    ///     rest of the codebase — nocopy reshape VIEWS flowing through arithmetic, reductions,
    ///     fancy indexing, iteration and IO-ish consumers; KEEPORDER sort/stack outputs feeding
    ///     downstream ops; nonzero's shared-buffer views driving selection; read-only reduction
    ///     scalars as operands and as refused write targets; and the broadcast writeable
    ///     override behaving under further view algebra. Char and Decimal (no NumPy analog)
    ///     ride the same paths via self-consistency sweeps.
    /// </summary>
    [TestClass]
    public class LayoutParityIntegrationTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
            NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
            NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        // ================================================================ reshape views

        [TestMethod]
        public void ReshapeView_WriteThrough_BothDirections()
        {
            var big = np.arange(24).astype(NPTypeCode.Int64);
            var st = big.reshape(3, 8)[":, ::2"];
            var view = st.reshape(2, 6);          // nocopy strided view (byte strides (96,16))

            view.SetAtIndex(999L, 0);             // write via the view …
            st.GetInt64(0, 0).Should().Be(999L, "the view shares the slice's memory");
            big.GetInt64(0).Should().Be(999L, "…and the ultimate base sees it");

            st.SetAtIndex(-5L, 1);                // write via the slice …
            view.GetInt64(0, 1).Should().Be(-5L, "…and the view sees it");
        }

        [TestMethod]
        public void ReshapeView_WriteThrough_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var src = np.arange(24).astype(tc).reshape(3, 8)[":, ::2"];
                var view = src.reshape(2, 6);
                var one = NDArray.Scalar(1).astype(tc);
                view.SetAtIndex(one.GetAtIndex(0), 0);
                src.GetAtIndex(0).Should().Be(view.GetAtIndex(0),
                    $"{tc}: the nocopy reshape view must alias its source");
            }
        }

        [TestMethod]
        public void ReshapeView_ChainedViews_StillAliasTheBase()
        {
            var big = np.arange(24).astype(NPTypeCode.Int64);
            // slice -> nocopy reshape -> slice -> same-shape reshape: a 4-deep view chain
            var chain = big.reshape(3, 8)[":, ::2"].reshape(2, 6)["1:2"].reshape(1, 6);
            chain.flags.owndata.Should().BeFalse();
            chain.SetAtIndex(777L, 0);
            // chain[0,0] is st[1,2]-of-(2,6) row 1 start == big flat 12
            big.GetInt64(12).Should().Be(777L, "a 4-deep view chain still writes the base buffer");
        }

        [TestMethod]
        public void ReshapeView_OpsSeeTheSameValuesAsAMaterializedCopy()
        {
            var srcs = new[]
            {
                np.arange(24).astype(NPTypeCode.Int64).reshape(3, 8)[":, ::2"].reshape(2, 6),
                np.arange(12).astype(NPTypeCode.Int64)["::-1"].reshape(3, 4),
                np.arange(20).astype(NPTypeCode.Int64)["4:16"].reshape(2, 6),
            };
            foreach (var v in srcs)
            {
                var copy = v.copy('C');
                np.sum(v).GetInt64().Should().Be(np.sum(copy).GetInt64());
                np.mean(v).GetDouble(0).Should().Be(np.mean(copy).GetDouble(0));
                np.array_equal(np.sort(v), np.sort(copy)).Should().BeTrue();
                np.array_equal(v + 1, copy + 1).Should().BeTrue();
                np.array_equal(v.T, copy.T).Should().BeTrue();
                np.array_equal(v["0"], copy["0"]).Should().BeTrue();
                np.array_equal(v[v > 5], copy[copy > 5]).Should().BeTrue();
                np.array_equal(v.astype(NPTypeCode.Double), copy.astype(NPTypeCode.Double)).Should().BeTrue();
                v.ToString().Should().Be(copy.ToString(), "printing reads logical order");
            }
        }

        [TestMethod]
        public void ReshapeView_AsFancyIndexOperand()
        {
            // the negative-stride nocopy view used AS the index array of a gather
            var idx = np.arange(6).astype(NPTypeCode.Int64)["::-1"].reshape(2, 3); // [[5,4,3],[2,1,0]]
            var data = np.arange(10, 22).astype(NPTypeCode.Int64);
            var taken = np.take(data, idx);
            np.array_equal(taken, np.take(data, idx.copy('C'))).Should().BeTrue(
                "a strided reshape view must gather identically to its materialized copy");
            taken.GetInt64(0, 0).Should().Be(15L);
        }

        [TestMethod]
        public void ReshapeView_CopytoWritesTheBase()
        {
            var big = np.arange(24).astype(NPTypeCode.Int64);
            var view = big.reshape(3, 8)[":, ::2"].reshape(2, 6);
            np.copyto(view, np.full(new Shape(2, 6), 7, NPTypeCode.Int64));
            for (int r = 0; r < 3; r++)
                for (int col = 0; col < 8; col += 2)
                    big.GetInt64(r * 8 + col).Should().Be(7L, "copyto through the view reaches the base");
            big.GetInt64(1).Should().Be(1L, "odd columns are outside the view and untouched");
        }

        [TestMethod]
        public void ReshapeView_OfReadOnlySource_StaysReadOnly_AndRefusesWrites()
        {
            var bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3));
            var view = bc.reshape(2, 2, 3);       // stride-0 nocopy view of a read-only broadcast
            view.flags.writeable.Should().BeFalse();
            var act = () => view.SetAtIndex(9L, 0);
            act.Should().Throw<NumSharpException>().WithMessage("*read-only*");

            var diag = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4).diagonal();
            diag.reshape(3, 1).flags.writeable.Should().BeFalse("reshape of np.diagonal's read-only view inherits");
        }

        [TestMethod]
        public void ReshapeViewOfCopy_IsIndependent_AndRefusesResize()
        {
            var src = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4).T; // F-ish, nocopy fails for C flatten
            var res = src.reshape(12);
            res.flags.owndata.Should().BeFalse("NumPy: base is the internal copy");
            res.SetAtIndex(555L, 0);
            src.GetInt64(0, 0).Should().Be(0L, "the copy path must not alias the source");

            var act = () => res.resize(new Shape(6));
            act.Should().Throw<IncorrectShapeException>("a view of the internal copy does not own its data — NumPy refuses resize too");
        }

        [TestMethod]
        public void ReshapeView_SurvivesSourceWrapperCollection()
        {
            NDArray MakeView()
            {
                var big = np.arange(1000).astype(NPTypeCode.Int64);
                return big.reshape(10, 100)[":, ::2"].reshape(5, 100);
            }

            var view = MakeView();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            np.sum(view).GetInt64().Should().Be(Enumerable.Range(0, 1000).Where(i => i % 100 % 2 == 0).Sum(i => (long)i),
                "ARC must keep the base buffer alive through the view alone");
        }

        [TestMethod]
        public void ReshapeF_View_TracksItsInternalCopy_NotTheSource()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var rf = a.reshape(new Shape(2, 6), 'F');
            rf.flags.owndata.Should().BeFalse();
            rf.flags.f_contiguous.Should().BeTrue();
            // NumPy value order: F-read of a C (3,4) is 0,4,8,1,5,9,2,6,10,3,7,11
            rf.SetAtIndex(999L, 0);
            a.GetInt64(0, 0).Should().Be(0L, "the F-copy is internal — writes must not reach the source");
            rf.GetInt64(0, 0).Should().Be(999L);
        }

        // ================================================================ K-order sort/partition

        [TestMethod]
        public void SortK_TakeAlongArgsort_EqualsSort_OnExoticLayouts()
        {
            var srcs = new (string name, NDArray a)[]
            {
                ("F", np.asfortranarray(np.array(new long[,] { { 9, 1, 5, 3 }, { 2, 8, 4, 6 }, { 7, 0, 11, 10 } }))),
                ("T", np.array(new long[,] { { 9, 1, 5 }, { 3, 2, 8 }, { 4, 6, 7 }, { 0, 11, 10 } }).T),
                ("st", np.arange(24).astype(NPTypeCode.Int64)["::-1"].reshape(3, 8)[":, ::2"]),
                ("t3d", np.transpose(np.arange(24).astype(NPTypeCode.Int64)["::-1"].reshape(2, 3, 4), new int[] { 2, 0, 1 })),
            };
            foreach (var (name, a) in srcs)
            {
                var sorted = np.sort(a, axis: -1);
                var gathered = np.take_along_axis(a, np.argsort(a, axis: -1), -1);
                np.array_equal(sorted, gathered).Should().BeTrue(
                    $"{name}: take_along_axis(a, argsort(a)) must equal sort(a) whatever layout sort's K-copy picked");
            }
        }

        [TestMethod]
        public void SortK_ResultIsWriteable_AndInPlaceSortIsIdempotent()
        {
            var f = np.asfortranarray(np.array(new long[,] { { 9, 1, 5, 3 }, { 2, 8, 4, 6 }, { 7, 0, 11, 10 } }));
            var s = np.sort(f);
            s.flags.writeable.Should().BeTrue();
            var before = s.tobytes('C');
            s.sort();                              // in-place on the F-layout copy
            CollectionAssert.AreEqual(before, s.tobytes('C'), "sorting a sorted F-layout array must be a no-op");
            s.SetAtIndex(42L, 0);
            f.GetInt64(0, 0).Should().Be(9L, "sort returns an owned copy — writes never reach the input");
        }

        [TestMethod]
        public void SortK_DecimalAndChar_SelfConsistent()
        {
            // no NumPy analog dtypes ride the same K-copy + in-place machinery
            var dec = np.asfortranarray(np.array(new decimal[,] { { 3.5m, 1.25m, 2m }, { 9m, 0m, 4.75m } }));
            var sd = np.sort(dec);
            sd.flags.f_contiguous.Should().BeTrue("Decimal sort inherits the K-copy layout");
            sd.GetAtIndex<decimal>(0).Should().Be(1.25m);
            sd.GetAtIndex<decimal>(2).Should().Be(3.5m);

            var ch = np.asfortranarray(np.array(new char[,] { { 'c', 'a', 'b' }, { 'z', 'x', 'y' } }));
            var sc = np.sort(ch);
            sc.flags.f_contiguous.Should().BeTrue();
            sc.GetAtIndex<char>(0).Should().Be('a');
        }

        [TestMethod]
        public void PartitionK_InvariantHolds_ThroughDownstreamReductions()
        {
            var t3 = np.transpose(np.arange(24).astype(NPTypeCode.Int64)["::-1"].reshape(2, 3, 4), new int[] { 2, 0, 1 });
            var part = np.partition(t3, 1, axis: -1);
            // the kth column equals sort's kth column, and reductions over the neither-contig
            // partition output agree with its C copy
            np.array_equal(np.take(part, 1L, axis: 2), np.take(np.sort(t3, axis: -1), 1L, axis: 2))
                .Should().BeTrue("partition's anchor column is sort's column");
            np.sum(part).GetInt64().Should().Be(np.sum(t3).GetInt64(), "partition permutes, never changes the multiset");
        }

        // ================================================================ stack/concat perm outputs

        [TestMethod]
        public void StackPermOutput_ValuesAndDownstreamOps_MatchCContiguousTwin()
        {
            var f = np.asfortranarray(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4));
            var perm = np.stack(new[] { f, f }, axis: 0);                    // (96,8,24) neither-contig
            var ctwin = np.stack(new[] { f.copy('C'), f.copy('C') }, axis: 0); // C-contig

            perm.flags.c_contiguous.Should().BeFalse();
            np.array_equal(perm, ctwin).Should().BeTrue();
            np.array_equal(np.sum(perm, 0), np.sum(ctwin, 0)).Should().BeTrue();
            np.array_equal(np.sum(perm, 2), np.sum(ctwin, 2)).Should().BeTrue();
            np.array_equal(perm.reshape(24), ctwin.reshape(24)).Should().BeTrue("reshape of the perm output reads logical order");
            np.array_equal(perm.T, ctwin.T).Should().BeTrue();
            perm.ToString().Should().Be(ctwin.ToString());
        }

        [TestMethod]
        public void StackPermOutput_IsWriteable_AndSliceWritesLand()
        {
            var f = np.asfortranarray(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4));
            var perm = np.stack(new[] { f, f }, axis: 1);                    // (16,8,48)
            perm.SetAtIndex(500L, 0);
            perm.GetInt64(0, 0, 0).Should().Be(500L);
            var slice = perm["1"];
            np.copyto(slice, np.zeros(new Shape(2, 4), NPTypeCode.Int64));
            np.sum(perm["1"]).GetInt64().Should().Be(0L, "writes through a slice of the perm output must land");
            f.GetInt64(0, 0).Should().Be(0L, "the stack output never aliases its inputs — input untouched apart from its own value");
        }

        [TestMethod]
        public void UnstackOfStack_RoundTrips_OnPermLayout()
        {
            var f = np.asfortranarray(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4));
            var perm = np.stack(new[] { f, f * 2 }, axis: 0);
            var parts = np.unstack(perm);
            parts.Length.Should().Be(2);
            np.array_equal(parts[0], f).Should().BeTrue();
            np.array_equal(parts[1], f * 2).Should().BeTrue();
            // unstack children are views of the perm output: write-through
            parts[0].SetAtIndex(123L, 0);
            perm.GetInt64(0, 0, 0).Should().Be(123L);
        }

        [TestMethod]
        public void ConcatMixedDtype_IntoPermVote_PromotesAndMatches()
        {
            var f32 = np.arange(6).astype(NPTypeCode.Single).reshape(2, 3);
            var f = np.asfortranarray(np.arange(6).astype(NPTypeCode.Double).reshape(2, 3));
            var cat = np.concatenate(new[] { f32, f }, 0);
            cat.typecode.Should().Be(NPTypeCode.Double);
            for (int i = 0; i < 6; i++)
            {
                cat.GetDouble(i / 3, i % 3).Should().Be(i);
                cat.GetDouble(2 + i / 3, i % 3).Should().Be(i);
            }
        }

        // ================================================================ nonzero shared-buffer views

        [TestMethod]
        public void NonZeroViews_DriveFancyGetAndSet()
        {
            var a = (np.arange(12).reshape(3, 4) % 3).astype(NPTypeCode.Int64);
            var nz = np.nonzero(a);
            var picked = a[nz[0], nz[1]];
            picked.size.Should().Be(np.count_nonzero(a));
            for (long i = 0; i < picked.size; i++)
                picked.GetInt64(i).Should().NotBe(0L);

            a[nz[0], nz[1]] = np.zeros(new Shape(picked.size), NPTypeCode.Int64);
            np.count_nonzero(a).Should().Be(0L, "setting through the strided nonzero views must clear exactly the nonzero cells");
        }

        [TestMethod]
        public void NonZeroViews_WriteThroughTheSharedBuffer_NotTheSource()
        {
            var a = (np.arange(12).reshape(3, 4) % 3).astype(NPTypeCode.Int64);
            var before = a.tobytes('C');
            var nz = np.nonzero(a);
            long other = nz[1].GetInt64(0);
            nz[0].SetAtIndex(2L, 0);              // mutate the row-coordinate column
            CollectionAssert.AreEqual(before, a.tobytes('C'), "the multi-index buffer is its own storage — the source array is untouched");
            nz[1].GetInt64(0).Should().Be(other, "the two tuple entries are DISJOINT columns of the shared buffer");
            nz[0].GetInt64(0).Should().Be(2L);
        }

        [TestMethod]
        public void NonZero_CountAndTake_AgreeAcrossLayouts()
        {
            foreach (var build in new Func<NDArray>[]
                     {
                         () => (np.arange(12).reshape(3, 4) % 3).astype(NPTypeCode.Int64),
                         () => np.asfortranarray((np.arange(12) % 3).astype(NPTypeCode.Int64).reshape(3, 4)),
                         () => (np.arange(12) % 3).astype(NPTypeCode.Int64).reshape(3, 4)["::-1"],
                         () => np.broadcast_to(np.array(new long[] { 0, 1, 0 }), new Shape(4, 3)),
                     })
            {
                var a = build();
                var nz = np.nonzero(a);
                nz[0].size.Should().Be(np.count_nonzero(a), "nonzero count must equal count_nonzero on every layout");
                if (a.ndim == 1)
                    continue;
                // gather through the STRIDED index views == gather through materialized copies
                var viaViews = a[nz[0], nz[1]];
                var viaCopies = a[nz[0].copy('C'), nz[1].copy('C')];
                np.array_equal(viaViews, viaCopies).Should().BeTrue();
            }
        }

        [TestMethod]
        public void NonZero_CharAndDecimal_SelfConsistent()
        {
            var dec = np.array(new decimal[] { 0m, 1.5m, 0m, -2m });
            var nzd = np.nonzero(dec);
            nzd[0].size.Should().Be(2);
            nzd[0].GetInt64(0).Should().Be(1L);
            nzd[0].GetInt64(1).Should().Be(3L);
            nzd[0].flags.owndata.Should().BeFalse("Decimal rides the same shared-buffer representation");

            var ch = np.array(new char[] { '\0', 'a', '\0', 'b' });
            var nzc = np.nonzero(ch);
            nzc[0].size.Should().Be(2);
            nzc[0].GetInt64(1).Should().Be(3L);
        }

        [TestMethod]
        public void NonZero_AllDtypes_MatchCountNonZero()
        {
            foreach (var tc in AllDtypes)
            {
                var a = (np.arange(12) % 3).astype(tc).reshape(3, 4);
                var nz = np.nonzero(a);
                nz.Length.Should().Be(2);
                nz[0].size.Should().Be(np.count_nonzero(a), $"{tc}: nonzero vs count_nonzero");
                nz[0].flags.owndata.Should().BeFalse($"{tc}: shared multi-index buffer");
            }
        }

        [TestMethod]
        public void WhereOneArg_FeedsSelectionConsumers()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var w = np.where(a > 5);
            var vals = a[w[0], w[1]];
            vals.size.Should().Be(6);
            np.amin(vals).GetInt64().Should().Be(6L);
            // trim_zeros and compress-style flows keep working over the view-backed indices
            np.array_equal(np.trim_zeros(np.array(new long[] { 0, 0, 3, 4, 0 })), np.array(new long[] { 3, 4 }))
                .Should().BeTrue();
        }

        // ================================================================ read-only reduction scalars

        [TestMethod]
        public void ReductionScalars_RefuseEveryWritePath()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var scalars = new (string name, NDArray r)[]
            {
                ("sum", np.sum(a)), ("prod", np.prod(np.arange(3).astype(NPTypeCode.Int64))),
                ("mean", np.mean(a)), ("std", np.std(a)), ("var", np.@var(a)),
                ("amax", np.amax(a)), ("amin", np.amin(a)), ("median", np.median(a)),
                ("ptp", np.ptp(a)), ("trace", np.trace(a)),
                ("percentile", np.percentile(a, 50.0)), ("quantile", np.quantile(a, 0.5)),
                ("nansum", np.nansum(np.array(new double[] { 1, double.NaN }))),
                ("nanmean", np.nanmean(np.array(new double[] { 1, double.NaN, 3 }))),
                ("nanstd", np.nanstd(np.array(new double[] { 1, double.NaN, 3 }))),
                ("nanvar", np.nanvar(np.array(new double[] { 1, double.NaN, 3 }))),
                ("nanmedian", np.nanmedian(np.array(new double[] { 1, double.NaN, 3 }))),
                ("nanmax", np.nanmax(np.array(new double[] { 1, double.NaN, 3 }))),
                ("sum_ax0_1d", np.sum(np.arange(3).astype(NPTypeCode.Int64), 0)),
                ("argmax_ax0_1d", np.argmax(np.arange(3).astype(NPTypeCode.Int64), 0)),
            };
            foreach (var (name, r) in scalars)
            {
                r.ndim.Should().Be(0, name);
                r.flags.writeable.Should().BeFalse($"{name} is NumPy's read-only scalar");
                // write probes use the result's OWN dtype (a mismatched raw SetAtIndex asserts
                // on the type before the writeable gate in Debug builds)
                var own = r.GetAtIndex(0);
                var set = () => r.SetAtIndex(own, 0);
                set.Should().Throw<NumSharpException>(name).WithMessage("*read-only*");
                var cpt = () => np.copyto(r, NDArray.Scalar(own));
                cpt.Should().Throw<NumSharpException>(name);
                var fil = () => r.fill(own);
                fil.Should().Throw<NumSharpException>(name);
            }
        }

        [TestMethod]
        public void ReductionScalars_WorkAsOperands_Everywhere()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var s = np.sum(a);                                     // read-only 66

            (s + 1).GetInt64().Should().Be(67L);
            (s * s).GetInt64().Should().Be(66L * 66L);
            np.array_equal(np.add(s, np.arange(3)), np.arange(3) + 66).Should().BeTrue();
            (a + s).GetInt64(0, 0).Should().Be(66L, "a read-only scalar broadcasts as an operand");
            np.mean(s).GetDouble(0).Should().Be(66.0, "reducing a read-only scalar input works");
            np.sum(s).flags.writeable.Should().BeFalse("and yields the scalar contract again");

            s.copy().flags.writeable.Should().BeTrue("copy() of the scalar re-owns writable, like NumPy's np.array(scalar)");
            s.astype(NPTypeCode.Double).flags.writeable.Should().BeTrue();
            s.astype(NPTypeCode.Double).GetDouble(0).Should().Be(66.0);
            s.ToString().Should().Be("66");
            s.GetInt64().Should().Be(66L);

            // comparison / logical consumers
            (s > 65).GetBoolean(0).Should().BeTrue();
            np.array_equal(s, NDArray.Scalar(66L)).Should().BeTrue();
        }

        [TestMethod]
        public void ReductionScalars_OutParameter_StaysWriteableAndReceives()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var o = NDArray.Scalar(0L);
            var r = a.TensorEngine.ReduceAdd(a, null, false, null, o);
            ReferenceEquals(r, o).Should().BeTrue("out= returns the out instance");
            r.flags.writeable.Should().BeTrue("NumPy: out is returned writeable, not converted to a scalar");
            r.GetInt64().Should().Be(66L);
            r.SetAtIndex(0L, 0);                    // and stays a legal write target
            r.GetInt64().Should().Be(0L);
        }

        [TestMethod]
        public void ReductionScalars_KeepdimsArrays_StayWriteable()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            var kd = np.sum(a, true);
            kd.ndim.Should().Be(2);
            kd.flags.writeable.Should().BeTrue("keepdims yields an ARRAY, not a scalar");
            kd.SetAtIndex(1L, 0);
            kd.GetInt64(0, 0).Should().Be(1L);

            np.sum(NDArray.Scalar(5L), true).flags.writeable
                .Should().BeFalse("keepdims over a 0-d input is STILL NumPy's scalar (probed 2.4.2)");
        }

        [TestMethod]
        public void ReductionScalars_NestedCompositions_KeepTheContract()
        {
            var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
            // inner axis reduction (writeable array) -> outer full reduction (read-only scalar)
            var inner = np.sum(a, 0);
            inner.flags.writeable.Should().BeTrue();
            var outer = np.sum(inner);
            outer.flags.writeable.Should().BeFalse();
            outer.GetInt64().Should().Be(66L);

            // np.average composes over the marked mean without tripping on read-onlyness
            np.average(a, (int?)null).GetDouble(0).Should().Be(5.5);
            // np.cov-style flows: reductions of reductions of views
            np.std(np.sort(np.asfortranarray(a))).GetDouble(0).Should().Be(np.std(a).GetDouble(0));
        }

        // ================================================================ broadcast writeable override

        [TestMethod]
        public void BroadcastWriteableOverride_WritesLandInSharedCells()
        {
            var basis = np.arange(3).astype(NPTypeCode.Int64);
            var bc = np.broadcast_to(basis, new Shape(4, 3));
            bc.setflags(write: true);

            var slice = bc["1:3"];
            slice.flags.writeable.Should().BeTrue();
            slice.SetAtIndex(99L, 0);               // stride-0 write: lands in basis[0]
            basis.GetInt64(0).Should().Be(99L, "a write through the re-enabled broadcast view hits the shared base cell");
            bc.GetInt64(0, 0).Should().Be(99L, "…and is visible through every row of the broadcast");
            bc.GetInt64(3, 0).Should().Be(99L);
        }

        [TestMethod]
        public void BroadcastWriteableOverride_DoesNotLeakIntoFreshBroadcasts()
        {
            var bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3));
            bc.setflags(write: true);

            np.broadcast_to(bc, new Shape(2, 4, 3)).flags.writeable
                .Should().BeFalse("broadcast_to always clears — the override never leaks forward");
            np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3))["1:3"]
                .flags.writeable.Should().BeFalse("a plain broadcast's views stay read-only");
        }

        [TestMethod]
        public void BroadcastWriteableOverride_ReclearImmediatelyProtectsNewViews()
        {
            var bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3));
            bc.setflags(write: true);
            bc["1:3"].flags.writeable.Should().BeTrue();

            bc.setflags(write: false);
            bc["1:3"].flags.writeable.Should().BeFalse("views created after the re-clear recompute read-only");
            var act = () => bc["1:3"].SetAtIndex(1L, 0);
            act.Should().Throw<NumSharpException>();
        }

        // ================================================================ KEEPORDER copy/astype

        [TestMethod]
        public void CopyAstypeK_ExoticLayouts_RoundTripValuesAndFeedReductions()
        {
            var t3 = np.transpose(np.arange(24).astype(NPTypeCode.Int64).reshape(2, 3, 4), new int[] { 2, 0, 1 });
            var k = t3.copy('K');
            k.flags.owndata.Should().BeTrue();
            k.flags.c_contiguous.Should().BeFalse("KEEPORDER preserved the transpose's stride order");
            np.array_equal(k, t3).Should().BeTrue();
            np.sum(k).GetInt64().Should().Be(np.sum(t3).GetInt64());
            np.array_equal(np.sum(k, 1), np.sum(t3, 1)).Should().BeTrue();

            var cast = t3.astype(NPTypeCode.Double);      // astype default order is K
            cast.flags.c_contiguous.Should().BeFalse();
            np.array_equal(cast.astype(NPTypeCode.Int64), t3).Should().BeTrue("K-layout astype round-trips");

            var bck = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3)).copy('K');
            bck.flags.writeable.Should().BeTrue("a K-copy of a broadcast is a fresh owned array");
            bck.SetAtIndex(9L, 0);
            bck.GetInt64(0, 0).Should().Be(9L);
        }

        [TestMethod]
        public void AstypeK_CopyFalse_ReturnsSelfOnSatisfiedStridedSources()
        {
            var st = np.arange(24).astype(NPTypeCode.Int64).reshape(3, 8)[":, ::2"];
            var same = st.astype(NPTypeCode.Int64, copy: false);
            ReferenceEquals(same, st).Should().BeTrue(
                "order 'K' imposes no layout constraint, so a satisfied copy:false returns the input itself (NumPy semantics)");
        }
    }
}
