using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     <see cref="Postprocess.TopK"/>'s selection route (k at most half the axis: np.argpartition, then a sort of the
    ///     k) against ORT's own TopK operator on vocabulary-wide rows, and the inputs that route hands to the whole-axis
    ///     sort (ties at the k-th value, a NaN) or serves on another axis or side, against an independent ranking of the
    ///     same contract (descending or ascending, NaN above every number, equal values lower index first).
    /// </summary>
    [TestClass]
    public class PostprocessTopKSelectionTests : OnnxTestBase
    {
        /// <summary>Gaussian float32 logits, three standard deviations wide, from NumPy's seeded generator.</summary>
        /// <param name="rows">Rows.</param>
        /// <param name="cols">Columns (the ranked axis).</param>
        /// <param name="seed">The seed.</param>
        /// <returns>A new (rows, cols) float32 array.</returns>
        private static NDArray RandomLogits(int rows, int cols, int seed)
        {
            np.random.seed(seed);
            using NDArray g = np.random.randn(rows, cols);
            using NDArray scaled = g * 3.0;
            return scaled.astype(NPTypeCode.Single);
        }

        /// <summary>The ONNX TopK ranking of one slice, computed directly: NaN ranks above every number, equal values
        ///     (either zero sign included) keep the lower index first.</summary>
        /// <param name="slice">The slice's values as doubles (every tested dtype converts exactly).</param>
        /// <param name="k">How many.</param>
        /// <param name="largest">True for the largest first.</param>
        /// <returns>The k indices in rank order.</returns>
        private static long[] Ranking(double[] slice, int k, bool largest)
        {
            int[] order = Enumerable.Range(0, slice.Length).ToArray();
            Array.Sort(order, (a, b) =>
            {
                double x = slice[a], y = slice[b];
                bool xn = double.IsNaN(x), yn = double.IsNaN(y);
                int c = xn && yn ? 0 : xn ? 1 : yn ? -1 : x.CompareTo(y);
                if (largest)
                    c = -c;
                return c != 0 ? c : a.CompareTo(b);
            });
            return order.Take(k).Select(i => (long)i).ToArray();
        }

        /// <summary>Asserts TopK along the last axis of a 2-D array against <see cref="Ranking"/>, row by row: indices
        ///     exactly, values as the gathered input values (bit for bit), dtype kept, results fresh and C-contiguous.</summary>
        /// <param name="x">A 2-D input.</param>
        /// <param name="k">How many.</param>
        /// <param name="largest">True for the largest first.</param>
        private static void AssertRowsMatchRanking(NDArray x, int k, bool largest)
        {
            (NDArray values, NDArray indices) = Postprocess.TopK(x, k, largest: largest);
            using (values)
            using (indices)
            {
                values.typecode.Should().Be(x.typecode);
                indices.typecode.Should().Be(NPTypeCode.Int64);
                values.Shape.IsContiguous.Should().BeTrue();
                indices.Shape.IsContiguous.Should().BeTrue();
                int rows = (int)x.shape[0];
                using NDArray x64 = x.astype(NPTypeCode.Double);
                using NDArray v64 = values.astype(NPTypeCode.Double);
                for (int r = 0; r < rows; r++)
                {
                    using NDArray row = x64[r];
                    double[] slice = row.ToArray<double>();
                    long[] want = Ranking(slice, k, largest);
                    using NDArray got = indices[r];
                    got.ToArray<long>().Should().Equal(want, $"row {r}");
                    using NDArray gotValues = v64[r];
                    double[] gv = gotValues.ToArray<double>();
                    for (int j = 0; j < k; j++)
                        BitConverter.DoubleToInt64Bits(gv[j]).Should().Be(BitConverter.DoubleToInt64Bits(slice[want[j]]), $"row {r}, rank {j}");
                }
            }
        }

        /// <summary>
        ///     Vocabulary-wide rows (51,200 entries, as a GPT-2 decoder's logits) through the selection route agree with
        ///     ORT's TopK exactly: indices, and values bit for bit.
        /// </summary>
        [TestMethod]
        public void WideRows_MatchOrtsTopK()
        {
            using InferenceSession session = OpenSession("topk_f32");
            using NDArray x = RandomLogits(3, 51200, seed: 7);
            foreach (int k in new[] { 1, 5, 50 })
            {
                using NDArray kTensor = np.array(new[] { (long)k });
                IReadOnlyDictionary<string, NDArray> ort = session.Run(new Dictionary<string, NDArray> { { "X", x }, { "K", kTensor } });
                using NDArray ortValues = ort["Values"];
                using NDArray ortIndices = ort["Indices"];
                (NDArray values, NDArray indices) = Postprocess.TopK(x, k);
                using (values)
                using (indices)
                {
                    np.array_equal(indices, ortIndices).Should().BeTrue($"k={k}: indices");
                    BytesOf(values).Should().Equal(BytesOf(ortValues), $"k={k}: values");
                }
            }
        }

        /// <summary>
        ///     Ties at the k-th value (small integers stored as float32: every value appears hundreds of times), where
        ///     np.argpartition may keep any of the tied entries: TopK rebuilds the set from the k-th values with the lower
        ///     indices among the ties, and agrees with ORT's choice. Ties wholly inside the selected k keep argpartition's
        ///     set.
        /// </summary>
        [TestMethod]
        public void TiesAtTheKthValue_MatchOrtsTopK()
        {
            using InferenceSession session = OpenSession("topk_f32");
            float[] narrow = Enumerable.Range(0, 2 * 4096).Select(i => (float)(i * 7919 % 10)).ToArray();
            float[] insideTies = Enumerable.Range(0, 2 * 4096).Select(i => (float)(i * 7919 % 8192)).ToArray();
            foreach (int i in new[] { 100, 2000, 3000 })
                insideTies[i] = 9000f;   // three equal maxima in row 0, all inside a top-5
            foreach (float[] data in new[] { narrow, insideTies })
            {
                using NDArray flat = np.array(data);
                using NDArray x = flat.reshape(2, 4096);
                using NDArray kTensor = np.array(new[] { 5L });
                IReadOnlyDictionary<string, NDArray> ort = session.Run(new Dictionary<string, NDArray> { { "X", x }, { "K", kTensor } });
                using NDArray ortValues = ort["Values"];
                using NDArray ortIndices = ort["Indices"];
                (NDArray values, NDArray indices) = Postprocess.TopK(x, 5);
                using (values)
                using (indices)
                {
                    np.array_equal(indices, ortIndices).Should().BeTrue();
                    np.array_equal(values, ortValues).Should().BeTrue();
                }
            }
        }

        /// <summary>
        ///     A NaN among the scores of a largest-first call takes the whole-row sort, which ranks NaN above every number
        ///     (lower index first among NaNs), and k = 1 takes np.argmax, whose first NaN is the same answer; the
        ///     smallest-first call keeps the selection, which never selects a NaN while enough numbers exist, and a slice
        ///     holding fewer than k numbers (its k-th value a NaN) is sorted.
        /// </summary>
        [TestMethod]
        public void NaN_RanksAboveEveryNumber()
        {
            using NDArray x = RandomLogits(2, 4096, seed: 11);
            x.SetSingle(float.NaN, 0, 4000);
            x.SetSingle(float.NaN, 0, 12);
            x.SetSingle(float.NaN, 1, 77);
            AssertRowsMatchRanking(x, 5, largest: true);
            AssertRowsMatchRanking(x, 1, largest: true);
            AssertRowsMatchRanking(x, 5, largest: false);
            // Row 1 keeps only three numbers: its 5 smallest include two NaNs.
            using NDArray mostlyNaN = np.full(new Shape(2, 600), float.NaN, np.float32);
            mostlyNaN.SetSingle(3f, 1, 10);
            mostlyNaN.SetSingle(1f, 1, 500);
            mostlyNaN.SetSingle(2f, 1, 20);
            for (int i = 0; i < 600; i++)
                mostlyNaN.SetSingle(i % 7, 0, i);
            AssertRowsMatchRanking(mostlyNaN, 5, largest: false);
        }

        /// <summary>
        ///     The smallest side, other dtypes (int32 with ties, int64, float64, float16) and the selection's k bound (half
        ///     the axis: the route still serves it; one more takes the sort).
        /// </summary>
        [TestMethod]
        public void SmallestSide_OtherDtypes_AndTheKBound()
        {
            using NDArray logits = RandomLogits(3, 2048, seed: 3);
            AssertRowsMatchRanking(logits, 5, largest: false);
            AssertRowsMatchRanking(logits, 1024, largest: true);
            AssertRowsMatchRanking(logits, 1025, largest: true);
            using NDArray scaled = logits * 10f;
            using NDArray ints = scaled.astype(NPTypeCode.Int32);
            AssertRowsMatchRanking(ints, 5, largest: true);
            AssertRowsMatchRanking(ints, 5, largest: false);
            using NDArray longs = scaled.astype(NPTypeCode.Int64);
            AssertRowsMatchRanking(longs, 7, largest: true);
            using NDArray doubles = logits.astype(NPTypeCode.Double);
            AssertRowsMatchRanking(doubles, 5, largest: true);
            using NDArray halves = logits.astype(NPTypeCode.Half);
            AssertRowsMatchRanking(halves, 5, largest: true);
        }

        /// <summary>
        ///     Ranking along axis 0 of a (4096, 3) array: the route ranks a moved-axis view and moves the results back, as
        ///     fresh C-contiguous (k, 3) arrays equal to the transposed problem's answer.
        /// </summary>
        [TestMethod]
        public void OtherAxis_ResultsMovedBack()
        {
            using NDArray rows = RandomLogits(3, 4096, seed: 5);
            using NDArray transposed = rows.T;
            using NDArray columns = transposed.copy();   // (4096, 3), C-contiguous
            (NDArray values, NDArray indices) = Postprocess.TopK(columns, 5, axis: 0);
            using (values)
            using (indices)
            {
                indices.shape.Should().Equal(5, 3);
                indices.Shape.IsContiguous.Should().BeTrue();
                values.Shape.IsContiguous.Should().BeTrue();
                (NDArray rowValues, NDArray rowIndices) = Postprocess.TopK(rows, 5);
                using (rowValues)
                using (rowIndices)
                {
                    using NDArray wantIndices = rowIndices.T;
                    using NDArray wantValues = rowValues.T;
                    np.array_equal(indices, wantIndices).Should().BeTrue();
                    np.array_equal(values, wantValues).Should().BeTrue();
                }
            }
        }
    }
}
