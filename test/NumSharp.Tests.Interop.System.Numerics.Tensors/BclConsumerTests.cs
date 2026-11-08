using System;
using System.Linq;
using System.Numerics.Tensors;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Consumer-driven contract tests: the spans the bridge hands out are driven through REAL
    ///     <c>System.Numerics.Tensors</c> operations — <c>Tensor.Sum</c>, in-place <c>Tensor.Add</c> /
    ///     <c>Tensor.Multiply</c> into a destination span, <c>Fill</c>, <c>Clear</c>, <c>CopyTo</c>,
    ///     <c>Tensor.SequenceEqual</c> — the way a consumer of the package uses them. Those operations walk lengths and
    ///     strides with the BCL's own traversal code (not FlattenTo, not the indexer), read AND write, so a span whose
    ///     strides, offset or reach the bridge got wrong produces a wrong sum or a stray write here. Values are small
    ///     integers held in float64/int32/int64, so every sum is exact in any order.
    /// </summary>
    [TestClass]
    public class BclConsumerTests : TensorsTestBase
    {
        /// <summary>The views the consumer tests run over: strided, offset, transposed, Fortran, broadcast, 0-d and empty.</summary>
        /// <param name="m">A C-contiguous 4×5 base.</param>
        /// <returns>The views, by name (each shares <paramref name="m"/>'s buffer).</returns>
        private static (string Name, NDArray View)[] Views(NDArray m) => new (string, NDArray)[]
        {
            ("whole (4,5)", m),
            ("row slice m[1:3]", m["1:3"]),
            ("column slice m[:, 1:4]", m[":, 1:4"]),
            ("stepped m[::2, 1::2]", m["::2, 1::2"]),
            ("transposed m.T", m.T),
            ("transposed slice m.T[1:4, ::3]", m.T["1:4, ::3"]),
            ("Fortran copy", np.asfortranarray(m)),
            ("broadcast row m[2] -> (3,5)", np.broadcast_to(m["2"], new Shape(3, 5))),
            ("0-d m[3, 4]", m["3, 4"]),
            ("empty m[2:2]", m["2:2"]),
        };

        /// <summary>
        ///     <c>Tensor.Sum</c> over each exported span equals the sum of NumSharp's own elements, for float64, int32 and
        ///     int64 — the BCL's reduction reads every element of the span exactly once through its strides.
        /// </summary>
        [TestMethod]
        public void TensorSum_OverExportedSpans_EqualsNumSharpsElements()
        {
            using NDArray d = np.arange(20.0).reshape(4, 5);
            using NDArray i32 = np.arange(20).astype(NPTypeCode.Int32).reshape(4, 5);
            using NDArray i64 = np.arange(20L).reshape(4, 5);
            foreach (var (name, view) in Views(d))
            {
                using var h = view.AsTensorSpan<double>();
                Tensor.Sum(h.ReadOnlySpan).Should().Be(Oracle.Elements<double>(view).Sum(), $"float64 {name}");
            }
            foreach (var (name, view) in Views(i32))
            {
                using var h = view.AsTensorSpan<int>();
                Tensor.Sum(h.ReadOnlySpan).Should().Be(Oracle.Elements<int>(view).Sum(), $"int32 {name}");
            }
            foreach (var (name, view) in Views(i64))
            {
                using var h = view.AsTensorSpan<long>();
                Tensor.Sum(h.ReadOnlySpan).Should().Be(Oracle.Elements<long>(view).Sum(), $"int64 {name}");
            }
        }

        /// <summary>
        ///     An in-place <c>Tensor.Add(span, 100, span)</c> through each writeable exported view adds 100 to exactly the
        ///     view's elements of the shared base and leaves every other base element alone — the BCL's strided WRITE
        ///     path lands where NumSharp's view says it should, and nowhere else.
        /// </summary>
        [TestMethod]
        public void InPlaceAdd_ThroughExportedSpans_TouchesExactlyTheView()
        {
            int count;
            using (NDArray probe = np.arange(20.0).reshape(4, 5))
                count = Views(probe).Length;
            for (int k = 0; k < count; k++)
            {
                // A fresh base per view, so each view's writes are checked against an untouched base.
                using NDArray m = np.arange(20.0).reshape(4, 5);
                var (name, view) = Views(m)[k];
                if (!view.Shape.IsWriteable || name == "Fortran copy")
                    continue;   // a read-only (broadcast) view refuses a mutable span; the Fortran copy has its own buffer

                double[] before = Oracle.Elements<double>(m);
                double[] expected = ExpectedAfter(m, view, before, x => x + 100);
                using (var h = view.AsTensorSpan<double>())
                {
                    TensorSpan<double> span = h.Span;
                    Tensor.Add(h.ReadOnlySpan, 100.0, span);
                }
                Oracle.Elements<double>(m).Should().Equal(expected, $"Tensor.Add through {name}");
            }
        }

        /// <summary>
        ///     <c>Fill</c>, <c>Clear</c> and an in-place <c>Tensor.Multiply</c> through a strided column slice and a
        ///     transposed slice change exactly the view's elements of the base.
        /// </summary>
        [TestMethod]
        public void FillClearMultiply_ThroughStridedSpans_TouchExactlyTheView()
        {
            foreach (string key in new[] { ":, 1:4", "::2, 1::2" })
            {
                using NDArray m = np.arange(1.0, 21.0).reshape(4, 5);
                using NDArray view = m[key];
                double[] start = Oracle.Elements<double>(m);

                using (var h = view.AsTensorSpan<double>())
                    h.Span.Fill(-7.0);
                double[] filled = ExpectedAfter(m, view, start, _ => -7.0);
                Oracle.Elements<double>(m).Should().Equal(filled, $"Fill through m[{key}]");

                using (var h = view.AsTensorSpan<double>())
                    Tensor.Multiply(h.ReadOnlySpan, 3.0, h.Span);
                double[] multiplied = ExpectedAfter(m, view, filled, x => x * 3);
                Oracle.Elements<double>(m).Should().Equal(multiplied, $"Tensor.Multiply through m[{key}]");

                using (var h = view.AsTensorSpan<double>())
                    h.Span.Clear();
                double[] cleared = ExpectedAfter(m, view, multiplied, _ => 0.0);
                Oracle.Elements<double>(m).Should().Equal(cleared, $"Clear through m[{key}]");
            }

            using NDArray t = np.arange(1.0, 21.0).reshape(4, 5);
            using NDArray tv = t.T["1:4, ::3"];
            double[] t0 = Oracle.Elements<double>(t);
            using (var h = tv.AsTensorSpan<double>())
                h.Span.Fill(99.0);
            Oracle.Elements<double>(t).Should().Equal(ExpectedAfter(t, tv, t0, _ => 99.0), "Fill through m.T[1:4, ::3]");
        }

        /// <summary>
        ///     The BCL copies between exported spans of DIFFERENT layouts by logical index: a transposed source copied
        ///     into a fresh C-ordered array's span, and into a Fortran-ordered array's span, lands as the transposed
        ///     values in both.
        /// </summary>
        [TestMethod]
        public void CopyTo_BetweenExportedSpansOfDifferentLayouts()
        {
            using NDArray m = np.arange(12.0).reshape(3, 4);
            using NDArray source = m.T;                                   // (4,3), strided
            double[] want = Oracle.Elements<double>(source);

            using NDArray cOrdered = np.zeros(new Shape(4, 3), np.float64);
            using NDArray fOrdered = np.asfortranarray(np.zeros(new Shape(4, 3), np.float64));
            foreach (NDArray destination in new[] { cOrdered, fOrdered })
            {
                using var src = source.AsTensorSpan<double>();
                using var dst = destination.AsTensorSpan<double>();
                src.ReadOnlySpan.CopyTo(dst.Span);
                Oracle.Elements<double>(destination).Should().Equal(want, destination.Shape.IsContiguous ? "into C order" : "into Fortran order");
            }
        }

        /// <summary>
        ///     The BCL's own element-wise comparison, <c>Tensor.Equals(x, y)</c>, finds every accepted catalogued
        ///     layout's zero-copy span equal, element for element, to its independent <c>ToTensor</c> copy (float64 and
        ///     int32); for a dense span, <c>Tensor.SequenceEqual</c> agrees.
        /// </summary>
        /// <remarks>
        ///     <c>Tensor.SequenceEqual</c> is only a valid oracle when both sides have the same LAYOUT: on
        ///     System.Numerics.Tensors 10.0.0 it returns <c>false</c> for any strided span against a dense copy of the
        ///     same elements — the BCL's own <c>t.Slice(.., 1..3)</c> or <c>Tensor.PermuteDimensions</c> results
        ///     included (verified 2026-10-02) — so it compares strides as well as elements. <c>Tensor.Equals</c> and
        ///     <c>Tensor.Sum</c> honour strides.
        /// </remarks>
        [TestMethod]
        public void ElementwiseEquals_ZeroCopySpan_EqualsItsCopy_EveryLayout()
        {
            foreach (var (name, build) in Layouts.All(i => i * 1.5 - 4))
            {
                NDArray view = build();
                if (Oracle.HasNegativeStride(view))
                    continue;
                using var h = view.AsTensorSpan<double>();
                Tensor<double> copy = view.ToTensor<double>();
                Oracle.Flatten(Tensor.Equals(h.ReadOnlySpan, copy.AsReadOnlyTensorSpan())).Should().OnlyContain(same => same, $"float64 {name}");
                if (h.ReadOnlySpan.IsDense)
                    Tensor.SequenceEqual(h.ReadOnlySpan, copy.AsReadOnlyTensorSpan()).Should().BeTrue($"float64 {name} (dense)");
            }
            foreach (var (name, build) in Layouts.All(i => (int)(i * 37 - 100)))
            {
                NDArray view = build();
                if (Oracle.HasNegativeStride(view))
                    continue;
                using var h = view.AsTensorSpan<int>();
                Tensor<int> copy = view.ToTensor<int>();
                Oracle.Flatten(Tensor.Equals(h.ReadOnlySpan, copy.AsReadOnlyTensorSpan())).Should().OnlyContain(same => same, $"int32 {name}");
            }
        }

        /// <summary>
        ///     A BCL operation's RESULT tensor comes back into NumSharp intact: <c>Tensor.Multiply</c> and
        ///     <c>Tensor.Add</c> over exported spans of every view give new tensors that, imported with ToNDArray and
        ///     AsNDArray, equal NumSharp's own <c>view * 3 + 1</c>.
        /// </summary>
        [TestMethod]
        public void BclResults_ImportBack_EqualToNumSharpsArithmetic()
        {
            using NDArray m = np.arange(20.0).reshape(4, 5);
            foreach (var (name, view) in Views(m))
            {
                using var h = view.AsTensorSpan<double>();
                Tensor<double> product = Tensor.Multiply(h.ReadOnlySpan, 3.0);
                Tensor<double> result = Tensor.Add(product.AsReadOnlyTensorSpan(), 1.0);
                using NDArray expected = view * 3 + 1;
                double[] want = Oracle.Elements<double>(expected);
                using NDArray copied = result.ToNDArray();
                using NDArray viewed = result.AsNDArray();
                long[] shape = view.ndim == 0 ? new long[] { 1 } : view.shape;
                copied.shape.Should().Equal(shape, name);
                Oracle.Elements<double>(copied).Should().Equal(want, $"ToNDArray of the BCL result over {name}");
                Oracle.Elements<double>(viewed).Should().Equal(want, $"AsNDArray of the BCL result over {name}");
            }
        }

        /// <summary>
        ///     The expected state of <paramref name="baseArray"/> after applying <paramref name="f"/> to exactly the
        ///     elements <paramref name="view"/> addresses (and to nothing else): each view element's base position is
        ///     found by writing a marker through NumSharp's own setter and seeing where it lands.
        /// </summary>
        /// <param name="baseArray">The C-contiguous base the view shares.</param>
        /// <param name="view">The view.</param>
        /// <param name="before">The base's elements before the operation.</param>
        /// <param name="f">The operation applied to each element the view addresses.</param>
        /// <returns>The base's expected elements after the operation.</returns>
        private static double[] ExpectedAfter(NDArray baseArray, NDArray view, double[] before, Func<double, double> f)
        {
            var expected = (double[])before.Clone();
            for (long i = 0; i < view.size; i++)
            {
                double old = view.GetAtIndex<double>(i);
                view.SetAtIndex(double.NaN, i);   // mark: which base position does view element i live at?
                double[] now = Oracle.Elements<double>(baseArray);
                view.SetAtIndex(old, i);
                int position = Array.FindIndex(now, double.IsNaN);
                expected[position] = f(before[position]);
            }
            return expected;
        }
    }
}
