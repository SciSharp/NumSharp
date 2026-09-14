using System;
using System.Numerics;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/array-creation.md</c> (the "Array creation"
    ///     fundamentals article). Each test mirrors one documented snippet and asserts the behaviour
    ///     the page claims, so the doc cannot silently drift from the library. Observed values were
    ///     captured by running the snippets against this branch (NumPy 2.4.2 parity).
    /// </summary>
    [TestClass]
    public class FundamentalsArrayCreationDocTests
    {
        // ── 1. Converting .NET sequences ───────────────────────────────────────────────────────────

        [TestMethod]
        public void Sequences_RankFollowsNesting()
        {
            np.array(new[] { 1, 2, 3, 4 }).shape.Should().Equal(new long[] { 4 });
            np.array(new[,] { { 1, 2 }, { 3, 4 } }).shape.Should().Equal(new long[] { 2, 2 });
            np.array(new[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).shape
              .Should().Equal(new long[] { 2, 2, 2 });
        }

        [TestMethod]
        public void Sequences_DtypeInferredFromElementType()
        {
            np.array(new[] { 1, 2, 3 }).typecode.Should().Be(NPTypeCode.Int32);
            np.array(new[] { 1.0, 2.0 }).typecode.Should().Be(NPTypeCode.Double);
            np.array(new sbyte[] { -1, 0, 1 }).typecode.Should().Be(NPTypeCode.SByte);
            np.array(new[] { (Half)1, (Half)2 }).typecode.Should().Be(NPTypeCode.Half);
            np.array(new[] { new Complex(1, 2) }).typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Sequences_ExplicitDtypeOverload()
        {
            var f = np.array(new[] { 1, 2, 3 }, np.float64);
            f.typecode.Should().Be(NPTypeCode.Double);
            f.ToArray<double>().Should().Equal(1.0, 2.0, 3.0);

            np.array(new[] { 1, 2, 3 }, np.int8).typecode.Should().Be(NPTypeCode.SByte);
        }

        [TestMethod]
        public void Sequences_Int32VsInt64_Gotcha()
        {
            np.array(new[] { 0, 1, 2 }).typecode.Should().Be(NPTypeCode.Int32, "np.array(int[]) follows the .NET int type");
            np.arange(3).typecode.Should().Be(NPTypeCode.Int64, "np.arange integer default is int64 (NumPy 2.x)");
        }

        [TestMethod]
        public void Sequences_StrongArrayWrapsOnDowncast()
        {
            // A C# int[] is a STRONG array → downcast WRAPS (unlike a Python list, which raises).
            np.array(new[] { 127, 128, 129 }, np.int8).ToArray<sbyte>().Should().Equal((sbyte)127, (sbyte)-128, (sbyte)-127);
        }

        // ── 2. Intrinsic creation functions ────────────────────────────────────────────────────────

        [TestMethod]
        public void Intrinsic_1D_Ranges()
        {
            np.arange(10).ToArray<long>().Should().Equal(0L, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            np.arange(2, 10, dtype: np.float64).typecode.Should().Be(NPTypeCode.Double);

            np.linspace(1.0, 4.0, 6).ToArray<double>().Should().Equal(1.0, 1.6, 2.2, 2.8, 3.4, 4.0);
            np.logspace(0, 2, 3).ToArray<double>().Should().Equal(1.0, 10.0, 100.0);
            np.geomspace(1, 1000, 4).ToArray<double>().Should().Equal(1.0, 10.0, 100.0, 1000.0);
        }

        [TestMethod]
        public void Intrinsic_2D_Constructors()
        {
            np.eye(3).shape.Should().Equal(new long[] { 3, 3 });
            np.eye(3, 5).shape.Should().Equal(new long[] { 3, 5 });

            var d = np.diag(np.array(new[] { 1, 2, 3 }));
            d.shape.Should().Equal(new long[] { 3, 3 });
            d.GetInt32(0, 0).Should().Be(1);
            d.GetInt32(1, 1).Should().Be(2);
            d.GetInt32(2, 2).Should().Be(3);

            // 2-D diag → the diagonal, as a view
            np.diag(np.array(new[,] { { 1, 2 }, { 3, 4 } })).ToArray<int>().Should().Equal(1, 4);

            np.vander(np.array(new[] { 1, 2, 3 }), 3).shape.Should().Equal(new long[] { 3, 3 });
        }

        [TestMethod]
        public void Intrinsic_Diag2D_IsReadOnlyView()
        {
            // The doc calls this out explicitly: np.diag on a 2-D input is a read-only view.
            np.diag(np.array(new[,] { { 1, 2 }, { 3, 4 } })).Shape.IsWriteable.Should().BeFalse();
        }

        [TestMethod]
        public void Intrinsic_ND_Fills_DefaultFloat64()
        {
            np.zeros((2, 3)).typecode.Should().Be(NPTypeCode.Double);
            np.zeros((2, 3)).shape.Should().Equal(new long[] { 2, 3 });
            np.ones((2, 3, 2)).shape.Should().Equal(new long[] { 2, 3, 2 });
            np.full((2, 2), 7.5).ToArray<double>().Should().Equal(7.5, 7.5, 7.5, 7.5);
            np.zeros((2, 3), np.int32).typecode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Intrinsic_Indices_TakesIntArray()
        {
            // np.indices takes an int[] of dimensions (NOT a tuple).
            np.indices(new[] { 3, 3 }).shape.Should().Equal(new long[] { 2, 3, 3 });
        }

        // ── 3. Replicating, joining, mutating ──────────────────────────────────────────────────────

        [TestMethod]
        public void Replicate_SliceIsView_CopyIsIndependent()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var b = a["0:2"];
            b[":"] = b + 1;
            a.ToArray<int>().Should().Equal(new[] { 2, 3, 3, 4, 5, 6 }, "the slice is a view, write-through mutates a");

            var a2 = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var c = a2["0:2"].copy();
            c[":"] = 0;
            a2.ToArray<int>().Should().Equal(new[] { 1, 2, 3, 4, 5, 6 }, "the copy is independent");
        }

        [TestMethod]
        public void Replicate_Block4x4()
        {
            var A = np.ones((2, 2));
            var B = np.eye(2, 2);
            var C = np.zeros((2, 2));
            var D = np.diag(np.array(new[] { -3, -4 }));
            np.block(new object[] { new object[] { A, B }, new object[] { C, D } }).shape
              .Should().Equal(new long[] { 4, 4 });
        }

        // ── 5. From raw bytes ──────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void RawBytes_Frombuffer_ReinterpretsFloat32()
        {
            byte[] raw = new byte[16];
            Buffer.BlockCopy(new[] { 1f, 2f, 3f, 4f }, 0, raw, 0, 16);
            np.frombuffer(raw, np.float32).ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);
        }

        // ── 6. Random (seed parity) ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Random_ShapesAndSeedReproducibility()
        {
            np.random.seed(42);
            var r1 = np.random.rand(2, 3);
            r1.shape.Should().Equal(new long[] { 2, 3 });

            np.random.seed(42);
            var r2 = np.random.rand(2, 3);
            np.array_equal(r1, r2).Should().BeTrue("the same seed reproduces the same sequence");

            np.random.randn(2, 3).shape.Should().Equal(new long[] { 2, 3 });
        }

        // ── Common patterns ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Patterns_ZerosLike_And_Meshgrid()
        {
            var template = np.zeros((2, 3), np.int32);
            var like = np.zeros_like(template);
            like.shape.Should().Equal(new long[] { 2, 3 });
            like.typecode.Should().Be(NPTypeCode.Int32);

            var (xx, yy) = np.meshgrid(np.arange(3), np.arange(4));
            xx.shape.Should().Equal(new long[] { 4, 3 });
            yy.shape.Should().Equal(new long[] { 4, 3 });
        }
    }
}
