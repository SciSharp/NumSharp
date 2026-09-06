using System;
using System.Collections.Generic;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     np.ix_ (open mesh) and np.s_ / np.index_exp (reusable index tuples).
    ///     Every expectation below is real NumPy 2.4.2 output.
    /// </summary>
    [TestClass]
    public class np_ix_Test
    {
        // ------------------------------------------------------------------ ix_: shapes

        [TestMethod]
        public void Ix_TwoSequences_AreOrthogonal()
        {
            // np.ix_([1,3],[2,5]) -> (array([[1],[3]]), array([[2, 5]]))
            var g = np.ix_(new long[] { 1, 3 }, new long[] { 2, 5 });
            g.Length.Should().Be(2);
            g[0].Should().BeShaped(2, 1).And.BeOfValues(1, 3);
            g[1].Should().BeShaped(1, 2).And.BeOfValues(2, 5);
        }

        [TestMethod]
        public void Ix_SingleSequence_StaysOneDimensional()
        {
            // nd == 1, so the reshape target is (size,) — NOT (size, 1).
            var g = np.ix_(new long[] { 1, 2, 3 });
            g.Length.Should().Be(1);
            g[0].Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        [TestMethod]
        public void Ix_ThreeSequences_CycleTheNonUnitAxis()
        {
            var g = np.ix_(new long[] { 1, 2 }, new long[] { 3 }, new long[] { 4, 5, 6 });
            g[0].Should().BeShaped(2, 1, 1).And.BeOfValues(1, 2);
            g[1].Should().BeShaped(1, 1, 1).And.BeOfValues(3);
            g[2].Should().BeShaped(1, 1, 3).And.BeOfValues(4, 5, 6);
        }

        [TestMethod]
        public void Ix_NoArguments_IsAnEmptyMesh()
        {
            np.ix_().Length.Should().Be(0);
        }

        // ------------------------------------------------------------------ ix_: dtypes

        [TestMethod]
        public void Ix_BooleanSequence_IsAMask()
        {
            // A bool sequence is equivalent to passing its nonzero() indices.
            np.ix_(new[] { true, false, true }, new long[] { 2, 4 })[0].Should()
                .BeShaped(2, 1).And.BeOfValues(0, 2);
            np.ix_(new[] { true, true }, new[] { false, false, true, false, true })[1].Should()
                .BeShaped(1, 2).And.BeOfValues(2, 4);
        }

        [TestMethod]
        public void Ix_PreservesTheInputDtype()
        {
            // ix_ performs NO integer validation, so the dtype rides through untouched —
            // a float sequence only fails later, at the indexing call.
            np.ix_(np.array(new sbyte[] { 1, 2 }))[0].typecode.Should().Be(NPTypeCode.SByte);
            np.ix_(np.array(new double[] { 1.0, 2.0 }))[0].typecode.Should().Be(NPTypeCode.Double);
            np.ix_(np.array(new char[] { 'a', 'b' }))[0].typecode.Should().Be(NPTypeCode.Char);
            np.ix_(np.array(new decimal[] { 1m, 2m }))[0].typecode.Should().Be(NPTypeCode.Decimal);
        }

        [TestMethod]
        public void Ix_EmptyNonArraySequence_IsTypedAsIntp()
        {
            // NumPy explicitly casts an empty non-ndarray input to intp so it does not default
            // to float64. intp is int64 on every platform NumSharp targets.
            np.ix_(Array.Empty<long>())[0].Should().BeShaped(0);
            np.ix_(Array.Empty<long>())[0].typecode.Should().Be(NPTypeCode.Int64);
            var g = np.ix_(Array.Empty<long>(), new long[] { 1, 2 });
            g[0].Should().BeShaped(0, 1);
            g[0].typecode.Should().Be(NPTypeCode.Int64);
            g[1].Should().BeShaped(1, 2);
        }

        // ------------------------------------------------------------------ ix_: view semantics

        [TestMethod]
        public void Ix_ReturnsViews_ThatWriteThrough()
        {
            var src = np.array(new long[] { 1, 2, 3 });
            var one = np.ix_(src)[0];
            one.SetValue(99L, 0);
            src.GetValue<long>(0).Should().Be(99, "the reshape is a view, as it is in NumPy");

            var two = np.ix_(src, new long[] { 0 })[0];
            two.SetValue(42L, 0, 0);
            src.GetValue<long>(0).Should().Be(42);
        }

        [TestMethod]
        public void Ix_OfAReadOnlyBroadcast_StaysReadOnly()
        {
            // NumPy: np.ix_(np.broadcast_to(...))[0].flags.writeable is False.
            var b = np.broadcast_to(NDArray.Scalar(7L), new Shape(4));
            np.ix_(b)[0].Shape.IsWriteable.Should().BeFalse();
            var g = np.ix_(b, new long[] { 1, 2 })[0];
            g.Shape.IsWriteable.Should().BeFalse();
            g.Should().BeShaped(4, 1);
        }

        [TestMethod]
        public void Ix_OfAStridedView_StaysStrided()
        {
            var s = np.arange(10)["::2"];
            var g = np.ix_(s, new long[] { 1 })[0];
            g.Should().BeShaped(5, 1).And.BeOfValues(0, 2, 4, 6, 8);
        }

        // ------------------------------------------------------------------ ix_: usage & errors

        [TestMethod]
        public void Ix_IndexesTheCrossProduct()
        {
            // a[np.ix_([0,1],[2,4])] -> [[2, 4], [7, 9]]
            var a = np.arange(10).reshape(2L, 5L);
            a[np.ix_(new long[] { 0, 1 }, new long[] { 2, 4 })].Should()
                .BeShaped(2, 2).And.BeOfValues(2, 4, 7, 9);
            a[np.ix_(new[] { true, true }, new long[] { 2, 4 })].Should()
                .BeShaped(2, 2).And.BeOfValues(2, 4, 7, 9);
        }

        [TestMethod]
        public void Ix_NonOneDimensionalSequence_Throws()
        {
            new Action(() => np.ix_(np.array(new long[,] { { 1, 2 } })))
                .Should().Throw<ValueError>().WithMessage("Cross index must be 1 dimensional*");
            new Action(() => np.ix_(NDArray.Scalar(5L)))
                .Should().Throw<ValueError>().WithMessage("Cross index must be 1 dimensional*");
        }

        [TestMethod]
        public void Ix_AcceptsCSharpCollections()
        {
            np.ix_(new List<int> { 1, 2 })[0].Should().BeShaped(2).And.BeOfValues(1, 2);
            np.ix_((1, 2))[0].Should().BeShaped(2).And.BeOfValues(1, 2);
        }

        [TestMethod]
        public void Ix_NullSequence_ThrowsTheSameValueErrorAsNumPy()
        {
            // NumPy arrives here by a different route — asarray(None) is a 0-d object array, whose
            // ndim is 0 — but the observable answer is this ValueError, not a bare
            // ArgumentNullException from inside the conversion helper.
            new Action(() => np.ix_((object)null))
                .Should().Throw<ValueError>().WithMessage("Cross index must be 1 dimensional*");
            new Action(() => np.ix_(new long[] { 1 }, null))
                .Should().Throw<ValueError>().WithMessage("Cross index must be 1 dimensional*");
        }

        [TestMethod]
        public void Ix_HighArity_BuildsTheFullRankMesh()
        {
            // The output rank equals the sequence count, so a 32-way mesh is 32-D. NumPy caps at
            // 64 dimensions; NumSharp has no such cap anywhere, so this is only checked for
            // structural sanity (each operand keeps its own non-unit axis).
            var seqs = new object[32];
            for (int i = 0; i < seqs.Length; i++)
                seqs[i] = new long[] { 0 };

            var mesh = np.ix_(seqs);
            mesh.Length.Should().Be(32);
            mesh[0].ndim.Should().Be(32);
            mesh[31].ndim.Should().Be(32);
        }

        // ------------------------------------------------------------------ s_ / index_exp

        [TestMethod]
        public void S_BuildsAReusableIndex()
        {
            // np.array([0,1,2,3,4])[np.s_[2::2]] -> array([2, 4])
            var every2nd = np.s_["2::2"];
            np.array(new long[] { 0, 1, 2, 3, 4 })[every2nd].Should()
                .BeShaped(2).And.BeOfValues(2, 4);

            // The same index object applies to a different array — that is the point of s_.
            var reversed = np.s_["::-1"];
            np.arange(4)[reversed].Should().BeOfValues(3, 2, 1, 0);
            np.arange(6)[reversed].Should().BeOfValues(5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void S_HandlesMultiAxisIndices()
        {
            var a = np.arange(12).reshape(3L, 4L);
            a[np.s_["1:, ::2"]].Should().BeShaped(2, 2).And.BeOfValues(4, 6, 8, 10);
            a[np.s_["..., -1"]].Should().BeShaped(3).And.BeOfValues(3, 7, 11);
            np.arange(3)[np.s_["np.newaxis, :"]].Should().BeShaped(1, 3);
        }

        [TestMethod]
        public void S_ComposesFromSliceObjects()
        {
            var a = np.arange(12).reshape(3L, 4L);
            a[np.s_[new Slice(1, 3), Slice.All]].Should()
                .BeShaped(2, 4).And.BeOfValues(4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void IndexExp_MatchesS_BecauseCSharpHasNoScalarVsTupleIndex()
        {
            // NumPy's s_ yields a bare slice and index_exp a 1-tuple; NDArray's indexer is
            // this[params Slice[]], so the two spellings are the same call in C#.
            var arr = np.array(new long[] { 0, 1, 2, 3, 4 });
            arr[np.index_exp["2::2"]].Should().BeOfValues(2, 4);
            arr[np.s_["2::2"]].Should().BeOfValues(2, 4);
            np.s_.maketuple.Should().BeFalse();
            np.index_exp.maketuple.Should().BeTrue();
        }

        [TestMethod]
        public void S_MalformedNotation_Throws()
        {
            new Action(() => _ = np.s_["bogus!"]).Should().Throw<ArgumentException>();
        }

        // ------------------------------------------------- s_: the full arr[...] vocabulary

        [TestMethod]
        public void S_CapturesAdvancedIndices_NotJustSlices()
        {
            // NumPy's s_ passes ANY index object through (np.s_[[1,2]] is the list), so every form
            // NDArray's indexer accepts must be expressible here too — fancy arrays and masks
            // included, not only the basic slice vocabulary.
            var a = np.arange(12).reshape(3L, 4L);
            var v = np.arange(5);

            a[np.s_[np.array(new[] { 0, 2 })]].Should()
                .BeShaped(2, 4).And.BeOfValues(0, 1, 2, 3, 8, 9, 10, 11);
            a[np.s_[new int[] { 0, 2 }]].Should().BeShaped(2, 4);
            a[np.s_[new long[] { 0, 2 }]].Should().BeShaped(2, 4);
            a[np.s_[new List<int> { 0, 2 }]].Should().BeShaped(2, 4);
            v[np.s_[new[] { true, false, true, false, true }]].Should()
                .BeShaped(3).And.BeOfValues(0, 2, 4);
        }

        [TestMethod]
        public void S_CapturesMixedBasicAndAdvancedIndices()
        {
            var a = np.arange(12).reshape(3L, 4L);
            a[np.s_[Slice.All, new int[] { 0, 2 }]].Should()
                .BeShaped(3, 2).And.BeOfValues(0, 2, 4, 6, 8, 10);
            a[np.s_["1:3", np.array(new[] { 0, 2 })]].Should()
                .BeShaped(2, 2).And.BeOfValues(4, 6, 8, 10);
            a[np.s_[np.ix_(new long[] { 0, 2 }, new long[] { 1, 3 })]].Should()
                .BeShaped(2, 2).And.BeOfValues(1, 3, 9, 11);
        }

        [TestMethod]
        public void S_BasicFormsStillReturnSliceArray_AdvancedReturnObjectArray()
        {
            // Overload resolution must keep the basic vocabulary on the Slice[] overload: a lone
            // Slice, an int (implicitly convertible to Slice) and a slice string all stay there.
            // The object[] overload only takes over when an entry cannot be a Slice.
            np.s_["1:3, ::2"].Should().BeOfType<Slice[]>();
            np.s_[new Slice(1, 3), Slice.All].Should().BeOfType<Slice[]>();
            np.s_[0].Should().BeOfType<Slice[]>();
            np.s_["1:3", "::2"].Should().BeOfType<Slice[]>();

            np.s_[np.array(new[] { 0, 2 })].Should().BeOfType<object[]>();
            np.s_[new int[] { 0, 2 }].Should().BeOfType<object[]>();
            np.s_[Slice.All, np.array(new[] { 0, 2 })].Should().BeOfType<object[]>();
        }

        [TestMethod]
        public void S_IsEquivalentToIndexingDirectly_AcrossTheWholeVocabulary()
        {
            // The contract: arr[X] and arr[np.s_[X]] are the same call for every X.
            var a = np.arange(12).reshape(3L, 4L);
            var v = np.arange(5);
            var fancy = np.array(new[] { 0, 2 });
            var mask = np.array(new[] { true, false, true, false, true });

            Same(a["1:3, ::2"], a[np.s_["1:3, ::2"]]);
            Same(a[new Slice(1, 3), Slice.All], a[np.s_[new Slice(1, 3), Slice.All]]);
            Same(a[0], a[np.s_[0]]);
            Same(a[fancy], a[np.s_[fancy]]);
            Same(a[new int[] { 0, 2 }], a[np.s_[new int[] { 0, 2 }]]);
            Same(v[mask], v[np.s_[mask]]);
            Same(a[Slice.All, new int[] { 0, 2 }], a[np.s_[Slice.All, new int[] { 0, 2 }]]);
            Same(a[Slice.NewAxis, Slice.All, Slice.All], a[np.s_[Slice.NewAxis, Slice.All, Slice.All]]);
            Same(a[Slice.Ellipsis, Slice.Index(0)], a[np.s_[Slice.Ellipsis, Slice.Index(0)]]);
        }

        private static void Same(NDArray direct, NDArray viaS_)
        {
            viaS_.typecode.Should().Be(direct.typecode);
            viaS_.shape.Should().BeEquivalentTo(direct.shape);
            viaS_.tobytes('C').Should().BeEquivalentTo(direct.tobytes('C'));
        }
    }
}
