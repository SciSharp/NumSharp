using System.Linq;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #284, "[Discussion] Ground Rules and Library Structure/Architecture"
    ///     (https://github.com/SciSharp/NumSharp/issues/284): NumPy 2.x is the reference (tested against NumPy 2.4.2), 15 dtypes
    ///     with complex128 among them, any ndim, computation in the <c>TensorEngine</c> with <c>np</c> as the API layer, and
    ///     NumPy's copy / view semantics.
    /// </summary>
    [TestClass]
    public class Issue284
    {
        /// <summary>
        ///     Every one of the 15 dtypes creates an array and adds in it, and complex128 arithmetic is complex
        ///     (<c>(1+2j) * (3-1j)</c> is <c>5+5j</c>).
        /// </summary>
        [TestMethod]
        public void FifteenDtypes_IncludingComplex128()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            };
            codes.Should().HaveCount(15);
            foreach (var code in codes)
            {
                var a = np.ones(new Shape(3), code);
                a.typecode.Should().Be(code);
                np.add(a, a).size.Should().Be(3, $"{code} computes");
            }

            var z = np.array(new[] { new System.Numerics.Complex(1, 2) }) * np.array(new[] { new System.Numerics.Complex(3, -1) });
            z.typecode.Should().Be(NPTypeCode.Complex);
            z.GetValue<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(5, 5));
        }

        /// <summary>Any ndim: a 32-dimensional array (NumPy 2.x allows 64) reshapes and reduces like any other.</summary>
        [TestMethod]
        public void AnyNdim_ThirtyTwoDimensions()
        {
            var dims = Enumerable.Repeat(1L, 31).Append(4L).ToArray();
            var a = np.arange(4).reshape(dims);

            a.ndim.Should().Be(32);
            np.sum(a).GetInt64().Should().Be(6);
        }

        /// <summary>
        ///     Computation lives in the <c>TensorEngine</c> and <c>np</c> is the API layer: <c>np.sum</c> is the engine's
        ///     <c>ReduceAdd</c>, and slicing returns NumPy's views (a write through the view changes the original).
        /// </summary>
        [TestMethod]
        public void EngineLayer_AndViewSemantics()
        {
            var a = np.arange(10);
            np.sum(a).GetInt64().Should().Be(a.TensorEngine.ReduceAdd(a, null, false).GetInt64());

            var view = a["2:5"];
            view[0] = 999;
            a.GetInt64(2).Should().Be(999, "slices are views, as in NumPy");
            var copy = a.copy();
            copy[0] = -1;
            a.GetInt64(0).Should().Be(0, "copy() is a copy");
        }

        /// <summary>The NumPy 2.4.2 oracle the answer cites is part of the repository: the differential-fuzz test project.</summary>
        [TestMethod]
        public void NumPyOracle_IsInTheRepository()
        {
            IssueAnswerSupport.ReadRepoFile("test/NumSharp.Tests.Oracle/NumSharp.Tests.Oracle.csproj").Should().NotBeNullOrWhiteSpace();
        }
    }
}
