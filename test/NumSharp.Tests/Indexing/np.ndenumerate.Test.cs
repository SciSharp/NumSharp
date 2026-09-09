using System;
using System.Globalization;
using System.Linq;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     Tests for <see cref="np.ndenumerate"/>. Every expected value comes from running
    ///     NumPy 2.4.2 (<c>list(np.ndenumerate(...))</c>) in the implementing session.
    /// </summary>
    [TestClass]
    public class np_ndenumerate_Test
    {
        private static string Fmt(np.NDEnumerate it)
            => string.Join(" ", it.Select(p =>
                "(" + string.Join(",", p.index) + ")=" +
                Convert.ToString(p.value, CultureInfo.InvariantCulture)));

        // ----------------------------------------------------------------- basic layouts

        [TestMethod]
        public void NdEnumerate_2D_COrder()
        {
            // numpy: [((0,0),1),((0,1),2),((1,0),3),((1,1),4)]
            var a = np.array(new[,] {{1, 2}, {3, 4}});
            Fmt(np.ndenumerate(a)).Should().Be("(0,0)=1 (0,1)=2 (1,0)=3 (1,1)=4");
        }

        [TestMethod]
        public void NdEnumerate_1D()
            => Fmt(np.ndenumerate(np.array(new[] {7, 8}))).Should().Be("(0)=7 (1)=8");

        [TestMethod]
        public void NdEnumerate_3D()
            // numpy: (0,0,0)=0 (0,0,1)=1 (0,1,0)=2 … (1,1,1)=7
            => Fmt(np.ndenumerate(np.arange(8).reshape(2, 2, 2)))
                .Should().Be("(0,0,0)=0 (0,0,1)=1 (0,1,0)=2 (0,1,1)=3 (1,0,0)=4 (1,0,1)=5 (1,1,0)=6 (1,1,1)=7");

        [TestMethod]
        public void NdEnumerate_ZeroD_YieldsOnePairWithEmptyIndex()
        {
            // numpy: list(np.ndenumerate(np.array(5))) == [((), 5)]
            var all = np.ndenumerate(NDArray.Scalar<int>(5)).ToArray();
            all.Should().HaveCount(1);
            all[0].index.Should().BeEmpty();
            all[0].value.Should().Be(5);
        }

        [TestMethod]
        public void NdEnumerate_Empty_YieldsNothing()
            // numpy: list(np.ndenumerate(np.zeros((0,3)))) == []
            => np.ndenumerate(np.zeros(new Shape(0, 3))).Should().BeEmpty();

        // ----------------------------------------------------------------- non-trivial memory layouts
        // ndenumerate is a flatiter walk: the order is ALWAYS logical C-order, whatever the layout.

        [TestMethod]
        public void NdEnumerate_FortranOrder_StillEnumeratesInCOrder()
        {
            var a = np.asfortranarray(np.array(new[,] {{1, 2}, {3, 4}}));
            Fmt(np.ndenumerate(a)).Should().Be("(0,0)=1 (0,1)=2 (1,0)=3 (1,1)=4");
        }

        [TestMethod]
        public void NdEnumerate_ReversedColumns_ReadsThroughNegativeStrides()
        {
            // numpy: a[:, ::-1] -> (0,0)=2 (0,1)=1 (1,0)=4 (1,1)=3
            var a = np.array(new[,] {{1, 2}, {3, 4}});
            Fmt(np.ndenumerate(a[":, ::-1"])).Should().Be("(0,0)=2 (0,1)=1 (1,0)=4 (1,1)=3");
        }

        [TestMethod]
        public void NdEnumerate_Transposed()
        {
            // numpy: a.T -> (0,0)=1 (0,1)=3 (1,0)=2 (1,1)=4
            var a = np.array(new[,] {{1, 2}, {3, 4}});
            Fmt(np.ndenumerate(a.T)).Should().Be("(0,0)=1 (0,1)=3 (1,0)=2 (1,1)=4");
        }

        [TestMethod]
        public void NdEnumerate_BroadcastView_RepeatsAlongStrideZeroAxis()
        {
            // numpy: broadcast_to([1,2],(2,2)) -> (0,0)=1 (0,1)=2 (1,0)=1 (1,1)=2
            var b = np.broadcast_to(np.array(new[] {1, 2}), new Shape(2, 2));
            Fmt(np.ndenumerate(b)).Should().Be("(0,0)=1 (0,1)=2 (1,0)=1 (1,1)=2");
        }

        [TestMethod]
        public void NdEnumerate_SlicedView_UsesTheViewsOwnShape()
        {
            // numpy: np.arange(12).reshape(3,4)[1:, ::2]
            //     -> (0,0)=4 (0,1)=6 (1,0)=8 (1,1)=10
            var a = np.arange(12).reshape(3, 4);
            Fmt(np.ndenumerate(a["1:, ::2"])).Should().Be("(0,0)=4 (0,1)=6 (1,0)=8 (1,1)=10");
        }

        // ----------------------------------------------------------------- dtypes

        [TestMethod]
        public void NdEnumerate_Float32()
            => Fmt(np.ndenumerate(np.array(new[] {1.5f, 2.5f}))).Should().Be("(0)=1.5 (1)=2.5");

        [TestMethod]
        public void NdEnumerate_Boolean()
            => Fmt(np.ndenumerate(np.array(new[] {true, false}))).Should().Be("(0)=True (1)=False");

        [TestMethod]
        public void NdEnumerate_Double()
            => Fmt(np.ndenumerate(np.array(new[] {0.5, -1.25}))).Should().Be("(0)=0.5 (1)=-1.25");

        // ----------------------------------------------------------------- typed (NumSharp extension)

        [TestMethod]
        public void NdEnumerate_Typed_MatchesUntyped()
        {
            var a = np.array(new[,] {{1, 2}, {3, 4}});
            string typed = string.Join(" ", np.ndenumerate<int>(a)
                .Select(p => "(" + string.Join(",", p.index) + ")=" + p.value));

            typed.Should().Be(Fmt(np.ndenumerate(a)));
        }

        [TestMethod]
        public void NdEnumerate_Typed_WrongElementType_Throws()
        {
            var a = np.array(new[] {1, 2});
            new Action(() => np.ndenumerate<double>(a).ToArray())
                .Should().Throw<ArgumentException>();
        }

        // ----------------------------------------------------------------- iterator identity

        [TestMethod]
        public void NdEnumerate_IsItsOwnIterator_SecondPassResumes()
        {
            // numpy: iter(e) is e
            var a = np.array(new[] {1, 2, 3, 4});
            var it = np.ndenumerate(a);

            it.Take(2).Select(p => p.value).Should().Equal(1, 2);
            it.Select(p => p.value).Should().Equal(3, 4);
        }

        [TestMethod]
        public void NdEnumerate_YieldsFreshIndexArrays()
        {
            var a = np.array(new[,] {{1, 2}, {3, 4}});
            var all = np.ndenumerate(a).ToArray();

            all.Select(p => string.Join(",", p.index))
               .Should().Equal("0,0", "0,1", "1,0", "1,1");
        }

        [TestMethod]
        public void NdEnumerate_IndicesAreIntp()
            => np.ndenumerate(np.array(new[] {1, 2})).First().index.Should().BeOfType<long[]>();

        // ----------------------------------------------------------------- cross-check against ndindex

        [TestMethod]
        public void NdEnumerate_IndicesMatchNdIndexOverTheSameShape()
        {
            var a = np.arange(24).reshape(2, 3, 4);

            np.ndenumerate(a).Select(p => string.Join(",", p.index))
                .Should().Equal(np.ndindex(a.shape).Select(i => string.Join(",", i)));
        }

        // -------------------------------------------- AsRef() — allocation-free, by-ref T, non-boxed

        private static string FmtRef(np.NDEnumerateRef<long> it)
        {
            var parts = new System.Collections.Generic.List<string>();
            foreach (var e in it)
                parts.Add("(" + string.Join(",", e.Index.ToArray()) + ")=" + e.Value);
            return string.Join(" ", parts);
        }

        [TestMethod]
        public void AsRef_MatchesBoxedOrder_AcrossLayouts()
        {
            var a = np.arange(6).reshape(2, 3);  // int64
            // Boxed reference strings (probed to equal NumPy's ndenumerate order).
            FmtRef(np.ndenumerate<long>(a).AsRef()).Should().Be("(0,0)=0 (0,1)=1 (0,2)=2 (1,0)=3 (1,1)=4 (1,2)=5");
            FmtRef(np.ndenumerate<long>(a.T).AsRef()).Should().Be("(0,0)=0 (0,1)=3 (1,0)=1 (1,1)=4 (2,0)=2 (2,1)=5");
            FmtRef(np.ndenumerate<long>(a[":, ::-1"]).AsRef()).Should().Be("(0,0)=2 (0,1)=1 (0,2)=0 (1,0)=5 (1,1)=4 (1,2)=3");
            FmtRef(np.ndenumerate<long>(a[":, ::2"]).AsRef()).Should().Be("(0,0)=0 (0,1)=2 (1,0)=3 (1,1)=5");
        }

        [TestMethod]
        public void AsRef_WriteThrough_CanUseIndex()
        {
            var a = np.arange(6).reshape(2, 3);
            foreach (var e in np.ndenumerate<long>(a).AsRef(writeable: true))
                e.Value = e.Value * 10 + e.Index[e.Index.Length - 1];  // fold the last coord into the value
            a.flatten().ToArray<long>().Should().Equal(0, 11, 22, 30, 41, 52);
        }

        [TestMethod]
        public void AsRef_ZeroD_YieldsOnePairEmptyIndex()
        {
            int count = 0, len = -1; long val = -1;
            foreach (var e in np.ndenumerate<long>(NDArray.Scalar(5L)).AsRef())
            {
                count++; len = e.Index.Length; val = e.Value;
            }
            count.Should().Be(1);
            len.Should().Be(0);
            val.Should().Be(5);
        }

        [TestMethod]
        public void AsRef_Empty_YieldsNothing()
        {
            int count = 0;
            foreach (var e in np.ndenumerate<long>(np.arange(0)).AsRef()) count++;
            count.Should().Be(0);
        }

        [TestMethod]
        public void AsRef_WrongElementType_Throws()
            => new Action(() =>
            {
                foreach (var e in np.ndenumerate<double>(np.arange(3)).AsRef()) { }
            }).Should().Throw<ArgumentException>();

        [TestMethod]
        public void AsRef_ReadOnlyBroadcast_Refuses_Writeable_But_Reads()
        {
            var bc = np.broadcast_to(np.arange(3), new Shape(2, 3));
            // read is fine (C-order values: the (3,) row repeats down the stride-0 axis)
            FmtRef(np.ndenumerate<long>(bc).AsRef()).Should().Be("(0,0)=0 (0,1)=1 (0,2)=2 (1,0)=0 (1,1)=1 (1,2)=2");
            // writeable refused with NumPy's read-only message
            new Action(() =>
            {
                foreach (var e in np.ndenumerate<long>(bc).AsRef(writeable: true)) { }
            }).Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void AsRef_ReEnumeration_Restarts()
        {
            // The typed value holds no cursor (unlike the boxed NDEnumerate, which resumes).
            var it = np.ndenumerate<long>(np.arange(4)).AsRef();
            FmtRef(it).Should().Be("(0)=0 (1)=1 (2)=2 (3)=3");
            FmtRef(it).Should().Be("(0)=0 (1)=1 (2)=2 (3)=3");
        }
    }
}
