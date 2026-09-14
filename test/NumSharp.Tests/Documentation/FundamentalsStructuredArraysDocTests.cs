using System;
using System.Linq;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples in
    ///     <c>docs/website-src/docs/fundamentals/structured-arrays.md</c> (the "Structured arrays"
    ///     fundamentals article). The article documents that NumSharp does NOT implement structured
    ///     dtypes; these tests pin that (the dtype strings throw) and the .NET-idiomatic replacements.
    /// </summary>
    [TestClass]
    public class FundamentalsStructuredArraysDocTests
    {
        // ── Not supported: structured dtype strings throw ──────────────────────────────────────────

        [TestMethod]
        public void StructuredDtypeStrings_Throw()
        {
            ((Action)(() => np.dtype("i8, f4, S3"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("3int8, float32"))).Should().Throw<NotSupportedException>();
        }

        // ── Replacement 1: parallel NDArrays (struct-of-arrays) ────────────────────────────────────

        [TestMethod]
        public void ParallelArrays_ColumnarReplacement()
        {
            string[] names = { "Rex", "Fido" };                  // text stays .NET
            var age = np.array(new[] { 9, 3 });                  // int32 column
            var weight = np.array(new[] { 81.0f, 27.0f });       // float32 column

            age.ToArray<int>().Should().Equal(9, 3);
            // numeric ops vectorize per column (what structured arrays are slow at)
            (age + 1).ToArray<int>().Should().Equal(10, 4);
            (weight * 2).ToArray<float>().Should().Equal(162f, 54f);

            // row 0 gathered across columns
            names[0].Should().Be("Rex");
            ((int)age[0]).Should().Be(9);
        }

        // ── Replacement 2: .NET record arrays ──────────────────────────────────────────────────────

        private record Pet(string Name, int Age, float Weight);

        [TestMethod]
        public void RecordArrays_BuildColumnFromField()
        {
            Pet[] pets = { new("Rex", 9, 81f), new("Fido", 3, 27f) };
            np.array(pets.Select(p => p.Weight).ToArray()).ToArray<float>().Should().Equal(81f, 27f);
        }
    }
}
