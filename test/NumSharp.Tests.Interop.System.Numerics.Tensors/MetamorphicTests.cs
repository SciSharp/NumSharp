using System;
using System.Buffers;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Metamorphic relations: instead of comparing one result against an oracle, two DIFFERENT routes to the same
    ///     answer must agree — slicing on the NumSharp side versus slicing on the BCL side, transposing before versus
    ///     after crossing, copying versus viewing, one round trip versus two. Each relation is run for all 15 dtypes, so
    ///     a route that is right for one element width and wrong for another breaks it.
    /// </summary>
    [TestClass]
    public class MetamorphicTests : TensorsTestBase
    {
        /// <summary>
        ///     Slicing then exporting equals exporting then slicing with the BCL: for every contiguous sub-block of a 3×4
        ///     array (and of its transpose), <c>nd[a:b, c:d].AsTensorSpan()</c> holds exactly the elements of
        ///     <c>nd.AsTensorSpan().ReadOnlySpan.Slice(a..b, c..d)</c>.
        /// </summary>
        [TestMethod]
        public void SliceThenExport_EqualsExportThenBclSlice()
        {
            var failures = new Failures();
            Dtypes.ForEach(new SliceExportRelation(failures));
            failures.AssertNone("slice/export");
        }

        /// <summary>
        ///     Importing then slicing equals slicing then importing: for every contiguous sub-block of a 3×4 tensor,
        ///     <c>t.Slice(a..b, c..d).AsNDArray()</c> and <c>.ToNDArray()</c> hold exactly the elements of
        ///     <c>t.AsNDArray()["a:b, c:d"]</c>.
        /// </summary>
        [TestMethod]
        public void SliceThenImport_EqualsImportThenNumSharpSlice()
        {
            var failures = new Failures();
            Dtypes.ForEach(new SliceImportRelation(failures));
            failures.AssertNone("slice/import");
        }

        /// <summary>
        ///     Transposing commutes with crossing, both ways and for every axis permutation of a 2×3×4 array: NumSharp's
        ///     <c>transpose(p)</c> then <c>ToTensor</c> equals <c>ToTensor</c> then the BCL's <c>PermuteDimensions(p)</c>,
        ///     and the BCL's permutation then <c>AsNDArray</c> equals <c>AsNDArray</c> then NumSharp's transpose.
        /// </summary>
        [TestMethod]
        public void Permuting_CommutesWithCrossing()
        {
            var failures = new Failures();
            Dtypes.ForEach(new PermuteRelation(failures));
            failures.AssertNone("permute");
        }

        /// <summary>
        ///     The copy and the zero-copy view of the same data agree: for every catalogued layout the exporter accepts,
        ///     <c>ToTensor</c> holds the same elements as <c>AsTensorSpan</c>; and two round trips equal one (the second
        ///     pass through ToTensor → ToNDArray / AsNDArray changes nothing).
        /// </summary>
        [TestMethod]
        public void CopyEqualsView_AndRoundTripsAreIdempotent()
        {
            var failures = new Failures();
            Dtypes.ForEach(new CopyViewRelation(failures));
            failures.AssertNone("copy/view");
        }

        /// <summary>
        ///     Zero-copy chains alias one buffer end to end: a tensor viewed by NumSharp and re-exported as a span is the
        ///     SAME memory — a write at one end is read at the other, for a dense, a strided and an offset tensor.
        /// </summary>
        [TestMethod]
        public void ZeroCopyChain_TensorToNDArrayToSpan_AliasesOneBuffer()
        {
            var failures = new Failures();
            Dtypes.ForEach(new AliasChainRelation(failures));
            failures.AssertNone("alias chain");
        }

        /// <summary>The contiguous step-1 ranges of an axis of length <paramref name="n"/>: every (start, stop) with start ≤ stop.</summary>
        /// <param name="n">The axis length.</param>
        /// <returns>The (start, stop) pairs.</returns>
        private static (int Start, int Stop)[] Ranges(int n)
            => (from a in Enumerable.Range(0, n + 1) from b in Enumerable.Range(a, n + 1 - a) select (a, b)).ToArray();

        /// <summary>Checks <see cref="SliceThenExport_EqualsExportThenBclSlice"/> for one dtype.</summary>
        private sealed class SliceExportRelation : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the relation check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public SliceExportRelation(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                NDArray m = Layouts.Base(value, 3, 4);
                foreach (bool transposed in new[] { false, true })
                {
                    NDArray source = transposed ? m.T : m;
                    long[] dims = source.shape;
                    foreach (var r in Ranges((int)dims[0]))
                        foreach (var c in Ranges((int)dims[1]))
                        {
                            string key = $"{r.Start}:{r.Stop}, {c.Start}:{c.Stop}";
                            _failures.Check($"{code} {(transposed ? "m.T" : "m")}[{key}]", () =>
                            {
                                using NDArray sliced = source[key];
                                using var viaNumSharp = sliced.AsTensorSpan<T>();
                                using var whole = source.AsTensorSpan<T>();
                                ReadOnlyTensorSpan<T> viaBcl = whole.ReadOnlySpan.Slice(new NRange[] { new NRange(r.Start, r.Stop), new NRange(c.Start, c.Stop) });
                                if (!Oracle.Longs(viaNumSharp.ReadOnlySpan.Lengths).SequenceEqual(Oracle.Longs(viaBcl.Lengths)))
                                    return "the two routes give different lengths";
                                return Oracle.SameBytes(Oracle.Flatten(viaNumSharp.ReadOnlySpan), Oracle.Flatten(viaBcl)) ? null : "the two routes hold different elements";
                            });
                        }
                }
            }
        }

        /// <summary>Checks <see cref="SliceThenImport_EqualsImportThenNumSharpSlice"/> for one dtype.</summary>
        private sealed class SliceImportRelation : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the relation check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public SliceImportRelation(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var backing = new T[12];
                for (int i = 0; i < backing.Length; i++)
                    backing[i] = value(i);
                Tensor<T> t = Tensor.Create(backing, new nint[] { 3, 4 });
                foreach (var r in Ranges(3))
                    foreach (var c in Ranges(4))
                    {
                        string key = $"{r.Start}:{r.Stop}, {c.Start}:{c.Stop}";
                        _failures.Check($"{code} t[{key}]", () =>
                        {
                            Tensor<T> slice = t.Slice(new NRange[] { new NRange(r.Start, r.Stop), new NRange(c.Start, c.Stop) });
                            using NDArray whole = t.AsNDArray();
                            using NDArray viaNumSharp = whole[key];
                            T[] want = Oracle.Elements<T>(viaNumSharp);
                            using NDArray view = slice.AsNDArray();
                            using NDArray copy = slice.ToNDArray();
                            if (!view.shape.SequenceEqual(viaNumSharp.shape) || !copy.shape.SequenceEqual(viaNumSharp.shape))
                                return "the routes give different shapes";
                            if (!Oracle.SameBytes(Oracle.Elements<T>(view), want))
                                return "AsNDArray of the slice differs from slicing the imported view";
                            return Oracle.SameBytes(Oracle.Elements<T>(copy), want) ? null : "ToNDArray of the slice differs from slicing the imported view";
                        });
                    }
            }
        }

        /// <summary>Checks <see cref="Permuting_CommutesWithCrossing"/> for one dtype.</summary>
        private sealed class PermuteRelation : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the relation check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public PermuteRelation(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                int[][] permutations =
                {
                    new[] { 0, 1, 2 }, new[] { 0, 2, 1 }, new[] { 1, 0, 2 },
                    new[] { 1, 2, 0 }, new[] { 2, 0, 1 }, new[] { 2, 1, 0 },
                };
                NDArray nd = Layouts.Base(value, 2, 3, 4);
                Tensor<T> t = nd.ToTensor<T>();
                foreach (int[] p in permutations)
                {
                    string perm = string.Join(",", p);
                    _failures.Check($"{code} export transpose({perm})", () =>
                    {
                        using NDArray transposed = nd.transpose(p);
                        Tensor<T> numSharpFirst = transposed.ToTensor<T>();
                        Tensor<T> bclFirst = Tensor.PermuteDimensions(t, p);
                        if (!Oracle.Longs(numSharpFirst.Lengths).SequenceEqual(Oracle.Longs(bclFirst.Lengths)))
                            return "the two routes give different lengths";
                        return Oracle.SameBytes(Oracle.Flatten(numSharpFirst), Oracle.Flatten(bclFirst)) ? null : "transpose-then-copy differs from copy-then-PermuteDimensions";
                    });
                    _failures.Check($"{code} import PermuteDimensions({perm})", () =>
                    {
                        using NDArray bclFirst = Tensor.PermuteDimensions(t, p).AsNDArray();
                        using NDArray imported = t.AsNDArray();
                        using NDArray numSharpFirst = imported.transpose(p);
                        if (!bclFirst.shape.SequenceEqual(numSharpFirst.shape))
                            return "the two routes give different shapes";
                        return Oracle.SameBytes(Oracle.Elements<T>(bclFirst), Oracle.Elements<T>(numSharpFirst)) ? null : "PermuteDimensions-then-view differs from view-then-transpose";
                    });
                }
            }
        }

        /// <summary>Checks <see cref="CopyEqualsView_AndRoundTripsAreIdempotent"/> for one dtype.</summary>
        private sealed class CopyViewRelation : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the relation check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public CopyViewRelation(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                foreach (var (name, build) in Layouts.All(value))
                {
                    NDArray nd = build();
                    if (!Oracle.HasNegativeStride(nd))
                        _failures.Check($"{code} {name} copy=view", () =>
                        {
                            using var h = nd.AsTensorSpan<T>();
                            return Oracle.SameBytes(Oracle.Flatten(nd.ToTensor<T>()), Oracle.Flatten(h.ReadOnlySpan)) ? null : "ToTensor differs from AsTensorSpan";
                        });
                    _failures.Check($"{code} {name} round trip idempotent", () =>
                    {
                        Tensor<T> once = nd.ToTensor<T>();
                        using NDArray back = once.ToNDArray();
                        Tensor<T> twice = back.ToTensor<T>();
                        using NDArray view = twice.AsNDArray();
                        Tensor<T> thrice = view.ToTensor<T>();
                        if (!Oracle.Longs(once.Lengths).SequenceEqual(Oracle.Longs(twice.Lengths)) || !Oracle.Longs(once.Lengths).SequenceEqual(Oracle.Longs(thrice.Lengths)))
                            return "a second or third round trip changed the lengths";
                        T[] first = Oracle.Flatten(once);
                        return Oracle.SameBytes(first, Oracle.Flatten(twice)) && Oracle.SameBytes(first, Oracle.Flatten(thrice)) ? null : "a second or third round trip changed the elements";
                    });
                }
            }
        }

        /// <summary>Checks <see cref="ZeroCopyChain_TensorToNDArrayToSpan_AliasesOneBuffer"/> for one dtype.</summary>
        private sealed class AliasChainRelation : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the relation check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public AliasChainRelation(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var backing = new T[20];
                for (int i = 0; i < backing.Length; i++)
                    backing[i] = value(i);
                Tensor<T> full = Tensor.Create(backing, new nint[] { 4, 5 });
                (string Name, Tensor<T> Tensor)[] tensors =
                {
                    ("dense (4,5)", full),
                    ("strided t[:, 1:4]", full.Slice(new NRange[] { NRange.All, new NRange(1, 4) })),
                    ("offset t[2:4, :]", full.Slice(new NRange[] { new NRange(2, 4), NRange.All })),
                    ("transposed", Tensor.PermuteDimensions(full, new[] { 1, 0 })),
                };
                foreach (var (name, t) in tensors)
                    _failures.Check($"{code} {name}", () =>
                    {
                        using NDArray view = t.AsNDArray();
                        using var h = view.AsTensorSpan<T>();
                        long[] lengths = Oracle.Longs(t.Lengths);
                        foreach (long[] index in Oracle.Indices(lengths))
                        {
                            nint[] at = Oracle.NInts(index);
                            T old = t[at];
                            T sentinel = Oracle.Different(old, value);
                            h.Span[at] = sentinel;                       // span end of the chain ...
                            bool tensorSees = Oracle.SameBytes(t[at], sentinel);
                            bool viewSees = Oracle.SameBytes(view.GetAtIndex<T>(Oracle.Flat(index, lengths)), sentinel);
                            t[at] = old;                                 // ... restored from the tensor end
                            bool spanRestored = Oracle.SameBytes(h.ReadOnlySpan[at], old);
                            if (!tensorSees || !viewSees || !spanRestored)
                                return $"index [{string.Join(",", index)}]: tensor sees {tensorSees}, NumSharp view sees {viewSees}, span sees the restore {spanRestored}";
                        }
                        return null;
                    });
            }
        }
    }
}
