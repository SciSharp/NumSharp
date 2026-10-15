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
    ///     Boundary-value analysis: inputs sitting exactly ON a limit and one step past it — the managed-array length
    ///     limit, the ends of a buffer, zero extents at every axis position, unit axes at every position, the highest
    ///     ranks — where off-by-one and wrong-limit defects live. Broadcast views stand in for the huge sizes, so no
    ///     test allocates more than a few kilobytes.
    /// </summary>
    [TestClass]
    public class BoundaryTests : TensorsTestBase
    {
        /// <summary>
        ///     <c>ToTensor</c> copies into a managed <c>T[]</c>, which holds at most <see cref="Array.MaxLength"/>
        ///     (2,147,483,591) elements — not <see cref="int.MaxValue"/>. One element past that limit, at
        ///     <see cref="int.MaxValue"/> and past it, the copy is refused with NotSupportedException pointing at
        ///     AsTensorSpan. Regression: the guard compared against <see cref="int.MaxValue"/>, so every size between the
        ///     two limits reached <c>new T[n]</c> and failed with OutOfMemoryException ("Array dimensions exceeded
        ///     supported range").
        /// </summary>
        [TestMethod]
        public void ToTensor_PastTheManagedArrayLimit_IsRefused_NotOutOfMemory()
        {
            foreach (long n in new[] { (long)Array.MaxLength + 1, int.MaxValue, (long)int.MaxValue + 1 })
            {
                using NDArray one = NDArray.Scalar(true);
                using NDArray huge = np.broadcast_to(one, new Shape(n));
                new Action(() => huge.ToTensor<bool>()).Should().Throw<NotSupportedException>($"{n} elements")
                    .WithMessage("*AsTensorSpan*");
            }
        }

        /// <summary>
        ///     The same sizes share zero-copy without limit: AsTensorSpan of a broadcast with
        ///     <see cref="Array.MaxLength"/> + 1 elements reports that length and reads its last element.
        /// </summary>
        [TestMethod]
        public void AsTensorSpan_PastTheManagedArrayLimit_SharesZeroCopy()
        {
            long n = (long)Array.MaxLength + 1;
            using NDArray one = NDArray.Scalar(2.5f);
            using NDArray huge = np.broadcast_to(one, new Shape(n));
            using var h = huge.AsTensorSpan<float>();
            ((long)h.ReadOnlySpan.FlattenedLength).Should().Be(n);
            h.ReadOnlySpan[new nint[] { (nint)(n - 1) }].Should().Be(2.5f);
        }

        /// <summary>
        ///     An empty view keeps its shape wherever it sits — at the start, mid-buffer, exactly at the end, reversed,
        ///     on either axis of a matrix, and in a transpose — through both export verbs and back, and the BCL can
        ///     reduce it. (An empty view can carry an offset at, or past, the last element: nothing may read there.)
        /// </summary>
        [TestMethod]
        public void EmptyViews_AtEveryPosition_KeepTheirShape()
        {
            using NDArray a = np.arange(8.0);
            using NDArray m = np.arange(20.0).reshape(4, 5);
            (string Key, NDArray View, long[] Dims)[] empties =
            {
                ("a[0:0]", a["0:0"], new long[] { 0 }),
                ("a[5:5]", a["5:5"], new long[] { 0 }),
                ("a[8:8]", a["8:8"], new long[] { 0 }),
                ("a[8:]", a["8:"], new long[] { 0 }),
                ("a[5:5:-1]", a["5:5:-1"], new long[] { 0 }),
                ("a[2:6:-1]", a["2:6:-1"], new long[] { 0 }),
                ("m[2:2]", m["2:2"], new long[] { 0, 5 }),
                ("m[:, 5:]", m[":, 5:"], new long[] { 4, 0 }),
                ("m[4:, 2:]", m["4:, 2:"], new long[] { 0, 3 }),
                ("m.T[5:]", m.T["5:"], new long[] { 0, 4 }),
            };
            foreach (var (key, view, dims) in empties)
            {
                using (view)
                {
                    view.shape.Should().Equal(dims, key);
                    using (var h = view.AsTensorSpan<double>())
                    {
                        Oracle.Longs(h.ReadOnlySpan.Lengths).Should().Equal(dims, $"{key}: span lengths");
                        Tensor.Sum(h.ReadOnlySpan).Should().Be(0.0, key);
                        using NDArray back = h.ReadOnlySpan.ToNDArray();
                        back.shape.Should().Equal(dims, $"{key}: span -> NDArray");
                    }
                    Tensor<double> t = view.ToTensor<double>();
                    Oracle.Longs(t.Lengths).Should().Equal(dims, $"{key}: ToTensor lengths");
                    using NDArray roundTrip = t.AsNDArray();
                    roundTrip.shape.Should().Equal(dims, $"{key}: ToTensor -> AsNDArray");
                }
            }
        }

        /// <summary>
        ///     A length-1 axis at EVERY position of ranks 1 to 4, stepped forwards or backwards: NumSharp gives the unit
        ///     axis a nonzero (possibly negative) stride, the BCL requires 0 there, and the bridge normalizes it — so
        ///     every such view exports, with every element and write in place (the full export contract per view).
        /// </summary>
        [TestMethod]
        public void UnitAxis_AtEveryPosition_EitherStepDirection_Exports()
        {
            var failures = new Failures();
            for (int rank = 1; rank <= 4; rank++)
                for (int unit = 0; unit < rank; unit++)
                    foreach (bool reversed in new[] { false, true })
                    {
                        long[] dims = Enumerable.Repeat(3L, rank).ToArray();
                        dims[unit] = 1;
                        NDArray view = Layouts.Base(i => i * 0.5, dims);
                        if (reversed)
                            view = view[string.Join(", ", Enumerable.Range(0, rank).Select(axis => axis == unit ? "::-1" : ":"))];
                        ExportContract.CheckAll(failures, $"{Oracle.Describe(dims)} unit axis {unit}{(reversed ? " reversed" : "")}", view, i => i * 0.5);
                    }
            failures.AssertNone("unit-axis view x verb");
        }

        /// <summary>
        ///     Views that reach the very last element of their buffer — through an offset (<c>a[-1:]</c>,
        ///     <c>a[9999:]</c>), through a long stride (<c>a[::997]</c> ends at element 9,970 of 10,000; <c>a[::1111]</c>
        ///     at 9,999 exactly) or through both axes of a matrix — export wholly: the span's reach covers the end and no
        ///     further.
        /// </summary>
        [TestMethod]
        public void ViewsReachingTheBufferEnd_ExportWholly()
        {
            var failures = new Failures();
            NDArray a = Layouts.Base(i => i * 0.25, 10_000);
            NDArray m = Layouts.Base(i => i * 0.25, 4, 5);
            (string Name, NDArray View)[] views =
            {
                ("a[-1:]", a["-1:"]),
                ("a[9999:]", a["9999:"]),
                ("a[::997]", a["::997"]),
                ("a[::1111]", a["::1111"]),
                ("a[1::1111]", a["1::1111"]),
                ("m[-1:, -1:]", m["-1:, -1:"]),
                ("m[1:, 2:]", m["1:, 2:"]),
                ("m.T[-1:]", m.T["-1:"]),
                ("m[::3, ::4]", m["::3, ::4"]),
            };
            foreach (var (name, view) in views)
                ExportContract.CheckAll(failures, name, view, i => i * 0.25);
            failures.AssertNone("buffer-end view x verb");
        }

        /// <summary>
        ///     The highest ranks cross both ways: rank-32 and rank-64 arrays (unit axes but three) through the full export
        ///     contract, and rank-64 tensors through both import verbs with their shape and elements intact.
        /// </summary>
        [TestMethod]
        public void HighRanks_CrossBothWays()
        {
            var failures = new Failures();
            foreach (int rank in new[] { 32, 64 })
            {
                long[] dims = Layouts.HighRank(rank);
                ExportContract.CheckAll(failures, $"rank {rank}", Layouts.Base(i => (long)i * 3, dims), i => i * 3);

                double[] values = Enumerable.Range(0, 12).Select(i => i + 0.5).ToArray();
                Tensor<double> t = Tensor.Create(values, Oracle.NInts(dims));
                failures.Check($"rank {rank} tensor / ToNDArray", () =>
                {
                    using NDArray copy = t.ToNDArray();
                    return ExportContract.Imported(copy, NPTypeCode.Double, dims, values);
                });
                failures.Check($"rank {rank} tensor / AsNDArray", () =>
                {
                    using NDArray view = t.AsNDArray();
                    return ExportContract.Imported(view, NPTypeCode.Double, dims, values);
                });
            }
            failures.AssertNone("high-rank");
        }

        /// <summary>
        ///     A zero extent at EVERY axis position of tensors of rank 1 to 4 imports with its shape through all three
        ///     import verbs, and the zero-copy one opens no lease (there is nothing to share).
        /// </summary>
        [TestMethod]
        public void ZeroExtent_AtEveryAxis_ImportsWithItsShape()
        {
            for (int rank = 1; rank <= 4; rank++)
                for (int zero = 0; zero < rank; zero++)
                {
                    long[] dims = Enumerable.Repeat(2L, rank).ToArray();
                    dims[zero] = 0;
                    Tensor<double> t = Tensor.Create(Array.Empty<double>(), Oracle.NInts(dims));
                    int leases = NDArrayTensorsInterop.LiveImports;
                    string name = Oracle.Describe(dims);
                    using (NDArray copy = t.ToNDArray())
                        copy.shape.Should().Equal(dims, $"{name} ToNDArray");
                    using (NDArray view = t.AsNDArray())
                    {
                        view.shape.Should().Equal(dims, $"{name} AsNDArray");
                        NDArrayTensorsInterop.LiveImports.Should().Be(leases, $"{name}: an empty tensor needs no lease");
                    }
                    using (NDArray span = t.AsReadOnlyTensorSpan().ToNDArray())
                        span.shape.Should().Equal(dims, $"{name} span.ToNDArray");
                }
        }
    }
}
