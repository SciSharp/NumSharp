using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Property-based and bounded-exhaustive coverage. Instead of hand-picked layouts, these tests GENERATE inputs —
    ///     seeded random NumSharp views (random rank, extents, slices with any step sign, integer indices, transposes,
    ///     expand_dims, broadcasts, read-only flags, in random order), seeded random <see cref="Tensor{T}"/>
    ///     constructions (dense, sub-blocks at a start offset, permuted, sliced, stride-0 broadcast, sliced-and-
    ///     permuted), and EVERY slice of a small vector / every axis-slice pair of a small matrix — and check invariants
    ///     that must hold for all of them, against NumSharp's or the BCL's own element reads.
    ///
    ///     <para>Every case's cell name carries the seed-derived recipe that built it, so a failure reproduces from the
    ///     message alone; the seeds are fixed, so the run is deterministic.</para>
    /// </summary>
    [TestClass]
    public class PropertyBasedTests : TensorsTestBase
    {
        /// <summary>The master seed; each dtype derives its own stream from it.</summary>
        private const int Seed = 20261002;

        /// <summary>Random views generated per dtype.</summary>
        private const int ViewsPerDtype = 120;

        /// <summary>Random tensors generated per dtype.</summary>
        private const int TensorsPerDtype = 120;

        /// <summary>
        ///     Every random view satisfies the export contract (<see cref="ExportContract.CheckAll{T}"/>): AsTensorSpan
        ///     refuses it exactly when it has a negative stride on an axis it steps, otherwise shares every element at the
        ///     right index and writes through to exactly the element named; ToTensor and the round trips reproduce
        ///     NumSharp's elements and shape.
        /// </summary>
        [TestMethod]
        public void RandomViews_SatisfyTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomViewCheck(failures));
            failures.AssertNone("random view x verb");
        }

        /// <summary>
        ///     Every random tensor imports exactly: ToNDArray, AsNDArray and span.ToNDArray reproduce its lengths and its
        ///     own elements (FlattenTo); the zero-copy view is writeable exactly when the tensor has no stride-0 axis;
        ///     a write through the view lands on the tensor element at the same multi-index, and vice versa; and
        ///     re-exporting the view aliases the tensor again.
        /// </summary>
        [TestMethod]
        public void RandomTensors_SatisfyTheImportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomTensorCheck(failures));
            failures.AssertNone("random tensor x verb");
        }

        /// <summary>
        ///     Bounded-exhaustive: EVERY start × stop × step slice of a length-5 vector (bounds -6..6 or omitted, steps
        ///     ±1/±2/±3/±5 or omitted — 1,764 views, every out-of-range clamp and empty result included) through the export
        ///     contract, for a 1-, an 8- and a 16-byte element type.
        /// </summary>
        [TestMethod]
        public void EverySliceOfAVector_SatisfiesTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new VectorSliceSweep(failures));
            failures.AssertNone("vector slice x verb");
        }

        /// <summary>
        ///     Bounded-exhaustive: every pair of per-axis slices (11 per axis: full, offset, stepped, reversed, empty,
        ///     integer index) of a 3×4 matrix, taken from the matrix or from its transpose, and each result also
        ///     transposed again — 484 views per dtype through the export contract.
        /// </summary>
        [TestMethod]
        public void EveryAxisSlicePairOfAMatrix_SatisfiesTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new MatrixSliceSweep(failures));
            failures.AssertNone("matrix slice x verb");
        }

        // ================================================================================== generators

        /// <summary>
        ///     One random per-axis slice token for an axis of length <paramref name="d"/>: an integer index (drops the
        ///     axis) one time in six, otherwise <c>start:stop:step</c> with a step of ±1, ±2 or ±3 and bounds that are
        ///     sometimes omitted.
        /// </summary>
        /// <param name="rng">The case's random stream.</param>
        /// <param name="d">The axis length.</param>
        /// <returns>The token, NumPy slice syntax.</returns>
        private static string AxisSlice(Random rng, long d)
        {
            if (d > 0 && rng.Next(6) == 0)
                return rng.NextInt64(-d, d).ToString();
            int step = new[] { 1, 2, 3, -1, -2, -3 }[rng.Next(6)];
            string start, stop;
            if (step > 0)
            {
                long a = rng.NextInt64(0, d + 1);
                long b = rng.NextInt64(a, d + 1);
                start = rng.Next(3) == 0 ? "" : a.ToString();
                stop = rng.Next(3) == 0 ? "" : b.ToString();
            }
            else
            {
                long a = d == 0 ? -1 : rng.NextInt64(0, d);   // the first element visited
                long b = rng.NextInt64(-1, a + 1);            // where to stop (-1: run through element 0)
                start = d == 0 || rng.Next(3) == 0 ? "" : a.ToString();
                stop = b < 0 || rng.Next(3) == 0 ? "" : b.ToString();
            }
            return $"{start}:{stop}:{step}";
        }

        /// <summary>
        ///     Builds a random NumSharp view: a C-contiguous base of rank 1-4 (an extent is 0 one time in twelve), then
        ///     zero to three of: slice every axis, transpose by a random permutation, expand_dims at a random axis,
        ///     broadcast_to a new leading axis with every unit axis stretched, or clear the writeable flag.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rng">The case's random stream.</param>
        /// <param name="value">The value generator.</param>
        /// <param name="recipe">Receives the steps taken, to reproduce a failure.</param>
        /// <returns>The view.</returns>
        private static NDArray RandomView<T>(Random rng, Func<long, T> value, StringBuilder recipe) where T : unmanaged
        {
            int rank = rng.Next(1, 5);
            var dims = new long[rank];
            for (int i = 0; i < rank; i++)
                dims[i] = rng.Next(12) == 0 ? 0 : rng.Next(1, 5);
            NDArray view = Layouts.Base(value, dims);
            recipe.Append("base").Append(Oracle.Describe(dims));

            int operations = rng.Next(0, 4);
            for (int k = 0; k < operations; k++)
            {
                switch (rng.Next(5))
                {
                    case 0:
                        if (view.ndim == 0)
                            break;
                        string key = string.Join(", ", view.shape.Select(d => AxisSlice(rng, d)));
                        view = view[key];
                        recipe.Append($" [{key}]");
                        break;
                    case 1:
                        if (view.ndim < 2)
                            break;
                        int[] perm = Enumerable.Range(0, view.ndim).OrderBy(_ => rng.Next()).ToArray();
                        view = view.transpose(perm);
                        recipe.Append($" .transpose({string.Join(",", perm)})");
                        break;
                    case 2:
                        int axis = rng.Next(0, view.ndim + 1);
                        view = np.expand_dims(view, axis);
                        recipe.Append($" expand_dims({axis})");
                        break;
                    case 3:
                        var target = new List<long> { rng.Next(1, 4) };
                        foreach (long d in view.shape)
                            target.Add(d == 1 ? rng.Next(1, 4) : d);
                        view = np.broadcast_to(view, new Shape(target.ToArray()));
                        recipe.Append($" broadcast_to{Oracle.Describe(target.ToArray())}");
                        break;
                    default:
                        view.setflags(write: false);
                        recipe.Append(" read-only");
                        break;
                }
            }
            return view;
        }

        /// <summary>A C-order strides vector (in elements) for a shape.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The strides.</returns>
        private static long[] CStrides(long[] dims)
        {
            var strides = new long[dims.Length];
            long s = 1;
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                strides[i] = s;
                s *= dims[i];
            }
            return strides;
        }

        /// <summary><paramref name="n"/> generated values.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The value generator.</param>
        /// <param name="n">How many.</param>
        /// <returns>The values, <paramref name="value"/>(0) first.</returns>
        private static T[] Values<T>(Func<long, T> value, long n)
        {
            var result = new T[n];
            for (long i = 0; i < n; i++)
                result[i] = value(i);
            return result;
        }

        /// <summary>A random permutation of 0..rank-1.</summary>
        /// <param name="rng">The case's random stream.</param>
        /// <param name="rank">The rank.</param>
        /// <returns>The permutation.</returns>
        private static int[] Permutation(Random rng, int rank) => Enumerable.Range(0, rank).OrderBy(_ => rng.Next()).ToArray();

        /// <summary>
        ///     Builds a random tensor of rank 1-4 (an extent is 0 one time in ten): dense; a sub-block of a padded array
        ///     at a start offset; a permuted view; a slice of a bigger tensor; a stride-0 broadcast on random axes; or a
        ///     slice that is then permuted. Length-1 axes get stride 0, as the BCL expects.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rng">The case's random stream.</param>
        /// <param name="value">The value generator.</param>
        /// <param name="recipe">Receives the construction, to reproduce a failure.</param>
        /// <returns>The tensor.</returns>
        private static Tensor<T> RandomTensor<T>(Random rng, Func<long, T> value, StringBuilder recipe)
        {
            int rank = rng.Next(1, 5);
            var lengths = new long[rank];
            for (int i = 0; i < rank; i++)
                lengths[i] = rng.Next(10) == 0 ? 0 : rng.Next(1, 5);
            recipe.Append(Oracle.Describe(lengths));
            if (lengths.Contains(0))
            {
                recipe.Append(" empty");
                return Tensor.Create(Array.Empty<T>(), Oracle.NInts(lengths));
            }

            switch (rng.Next(6))
            {
                case 0:
                    recipe.Append(" dense");
                    return Tensor.Create(Values(value, lengths.Aggregate(1L, (a, b) => a * b)), Oracle.NInts(lengths));

                case 1:
                {
                    long[] padded = lengths.Select(l => l + rng.Next(0, 3)).ToArray();
                    long[] cs = CStrides(padded);
                    long start = 0;
                    for (int i = 0; i < rank; i++)
                        start += rng.NextInt64(0, padded[i] - lengths[i] + 1) * cs[i];
                    nint[] strides = Enumerable.Range(0, rank).Select(i => lengths[i] <= 1 ? (nint)0 : (nint)cs[i]).ToArray();
                    recipe.Append($" sub-block of {Oracle.Describe(padded)} at start {start}");
                    T[] backing = Values(value, padded.Aggregate(1L, (a, b) => a * b));
                    return Tensor.Create(backing, (int)start, Oracle.NInts(lengths), strides);
                }

                case 2:
                {
                    int[] perm = Permutation(rng, rank);
                    var source = new long[rank];
                    for (int i = 0; i < rank; i++)
                        source[perm[i]] = lengths[i];
                    recipe.Append($" PermuteDimensions({string.Join(",", perm)}) of dense {Oracle.Describe(source)}");
                    var dense = Tensor.Create(Values(value, source.Aggregate(1L, (a, b) => a * b)), Oracle.NInts(source));
                    return Tensor.PermuteDimensions(dense, perm);
                }

                case 3:
                    return SlicedTensor(rng, value, lengths, recipe);

                case 4:
                {
                    bool[] broadcast = lengths.Select(l => l > 1 && rng.Next(3) == 0).ToArray();
                    long[] baseDims = Enumerable.Range(0, rank).Select(i => broadcast[i] ? 1L : lengths[i]).ToArray();
                    long[] cs = CStrides(baseDims);
                    nint[] strides = Enumerable.Range(0, rank).Select(i => broadcast[i] || lengths[i] <= 1 ? (nint)0 : (nint)cs[i]).ToArray();
                    recipe.Append($" stride-0 broadcast of {Oracle.Describe(baseDims)} on axes [{string.Join(",", Enumerable.Range(0, rank).Where(i => broadcast[i]))}]");
                    return Tensor.Create(Values(value, baseDims.Aggregate(1L, (a, b) => a * b)), Oracle.NInts(lengths), strides);
                }

                default:
                {
                    int[] perm = Permutation(rng, rank);
                    var source = new long[rank];
                    for (int i = 0; i < rank; i++)
                        source[perm[i]] = lengths[i];
                    Tensor<T> sliced = SlicedTensor(rng, value, source, recipe);
                    recipe.Append($" then PermuteDimensions({string.Join(",", perm)})");
                    return Tensor.PermuteDimensions(sliced, perm);
                }
            }
        }

        /// <summary>A slice of the given lengths taken at random offsets out of a dense tensor up to two larger per axis.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rng">The case's random stream.</param>
        /// <param name="value">The value generator.</param>
        /// <param name="lengths">The slice's lengths.</param>
        /// <param name="recipe">Receives the construction.</param>
        /// <returns>The sliced tensor (a view of the bigger one).</returns>
        private static Tensor<T> SlicedTensor<T>(Random rng, Func<long, T> value, long[] lengths, StringBuilder recipe)
        {
            long[] big = lengths.Select(l => l + rng.Next(0, 3)).ToArray();
            var source = Tensor.Create(Values(value, big.Aggregate(1L, (a, b) => a * b)), Oracle.NInts(big));
            var ranges = new NRange[lengths.Length];
            var offsets = new long[lengths.Length];
            for (int i = 0; i < lengths.Length; i++)
            {
                offsets[i] = rng.NextInt64(0, big[i] - lengths[i] + 1);
                ranges[i] = new NRange((nint)offsets[i], (nint)(offsets[i] + lengths[i]));
            }
            recipe.Append($" slice of dense {Oracle.Describe(big)} at {Oracle.Describe(offsets)}");
            return source.Slice(ranges);
        }

        // ================================================================================== the checks

        /// <summary>Generates <see cref="ViewsPerDtype"/> random views per dtype and runs the export contract on each.</summary>
        private sealed class RandomViewCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public RandomViewCheck(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var rng = new Random(Seed + (int)code * 7919);
                for (int c = 0; c < ViewsPerDtype; c++)
                {
                    var recipe = new StringBuilder();
                    NDArray view;
                    try
                    {
                        view = RandomView(rng, value, recipe);
                    }
                    catch (Exception e)
                    {
                        _failures.Add($"{code} view #{c} [{recipe}]", $"building it threw {e.GetType().Name}: {e.Message}");
                        continue;
                    }
                    ExportContract.CheckAll(_failures, $"{code} view #{c} [{recipe}]", view, value, rng);
                }
            }
        }

        /// <summary>Generates <see cref="TensorsPerDtype"/> random tensors per dtype and checks every import property.</summary>
        private sealed class RandomTensorCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public RandomTensorCheck(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var rng = new Random(Seed - (int)code * 104729);
                for (int c = 0; c < TensorsPerDtype; c++)
                {
                    var recipe = new StringBuilder();
                    Tensor<T> t;
                    try
                    {
                        t = RandomTensor(rng, value, recipe);
                    }
                    catch (Exception e)
                    {
                        _failures.Add($"{code} tensor #{c} [{recipe}]", $"building it threw {e.GetType().Name}: {e.Message}");
                        continue;
                    }
                    CheckImport(_failures, $"{code} tensor #{c} [{recipe}]", t, code, value, rng);
                }
            }
        }

        /// <summary>
        ///     The import contract for one tensor: every import verb reproduces the tensor's own elements and lengths,
        ///     the zero-copy view's writeability follows the tensor's strides, writes cross in both directions at the
        ///     same multi-index, and a re-export of the view aliases the tensor.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The tensor's name (dtype, case number, recipe).</param>
        /// <param name="t">The tensor.</param>
        /// <param name="code">The tensor's NumSharp dtype.</param>
        /// <param name="value">The value generator, for sentinels.</param>
        /// <param name="rng">The case's random stream (which element to write).</param>
        private static void CheckImport<T>(Failures failures, string cell, Tensor<T> t, NPTypeCode code, Func<long, T> value, Random rng) where T : unmanaged
        {
            T[] want = Oracle.Flatten(t);
            long[] lengths = Oracle.Longs(t.Lengths);
            long[] strides = Oracle.Longs(t.Strides);
            // A stride-0 axis means a broadcast — read-only in NumSharp — but only when an element exists to alias:
            // the BCL gives an EMPTY tensor all-zero strides, and the bridge imports it as a fresh, writeable empty array.
            bool writeable = want.Length == 0 || !Oracle.HasZeroStride(lengths, strides);

            failures.Check($"{cell} / ToNDArray", () =>
            {
                using NDArray copy = t.ToNDArray();
                return ExportContract.Imported(copy, code, lengths, want);
            });

            failures.Check($"{cell} / span.ToNDArray", () =>
            {
                using NDArray copy = t.AsReadOnlyTensorSpan().ToNDArray();
                return ExportContract.Imported(copy, code, lengths, want);
            });

            failures.Check($"{cell} / AsNDArray", () =>
            {
                using NDArray view = t.AsNDArray();
                string problem = ExportContract.Imported(view, code, lengths, want);
                if (problem != null)
                    return problem;
                if (view.Shape.IsWriteable != writeable)
                    return $"view writeable {view.Shape.IsWriteable}, but the tensor {(writeable ? "has no" : "has a")} stride-0 axis";
                if (!writeable || want.Length == 0)
                    return null;

                long flat = rng.NextInt64(want.Length);
                nint[] at = Oracle.NInts(Oracle.Unravel(flat, lengths));

                // NumSharp -> tensor: the view's flat element is the tensor's element at the same multi-index.
                T old = t[at];
                T sentinel = Oracle.Different(old, value);
                view.SetAtIndex(sentinel, flat);
                T[] after = Oracle.Flatten(t);
                view.SetAtIndex(old, flat);
                if (!Oracle.SameBytes(after[flat], sentinel))
                    return $"a write to view element {flat} is not visible at tensor[{string.Join(",", at.Select(x => (long)x))}]";
                for (long i = 0; i < after.LongLength; i++)
                    if (i != flat && !Oracle.SameBytes(after[i], want[i]))
                        return $"a write to view element {flat} also changed tensor element {i}";

                // tensor -> NumSharp.
                t[at] = sentinel;
                T seen = view.GetAtIndex<T>(flat);
                t[at] = old;
                return Oracle.SameBytes(seen, sentinel) ? null : $"a write to tensor[{string.Join(",", at.Select(x => (long)x))}] is not visible at view element {flat}";
            });

            failures.Check($"{cell} / AsNDArray->AsTensorSpan", () =>
            {
                using NDArray view = t.AsNDArray();
                using var h = view.AsTensorSpan<T>();
                if (!Oracle.SameBytes(Oracle.Flatten(h.ReadOnlySpan), want))
                    return "the re-exported span's elements differ from the tensor's";
                if (!h.IsWriteable || want.Length == 0)
                    return null;
                long flat = rng.NextInt64(want.Length);
                nint[] at = Oracle.NInts(Oracle.Unravel(flat, lengths));
                T old = t[at];
                T sentinel = Oracle.Different(old, value);
                h.Span[at] = sentinel;
                bool seen = Oracle.SameBytes(t[at], sentinel);
                h.Span[at] = old;
                return seen ? null : "a write through the re-exported span did not reach the tensor";
            });
        }

        /// <summary>Every slice of a length-5 vector, for byte, double and Complex.</summary>
        private sealed class VectorSliceSweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public VectorSliceSweep(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code != NPTypeCode.Byte && code != NPTypeCode.Double && code != NPTypeCode.Complex)
                    return;
                string[] bounds = new[] { "" }.Concat(Enumerable.Range(-6, 13).Select(i => i.ToString())).ToArray();
                string[] steps = { "", "1", "2", "3", "5", "-1", "-2", "-3", "-5" };
                NDArray vector = Layouts.Base(value, 5);
                foreach (string start in bounds)
                    foreach (string stop in bounds)
                        foreach (string step in steps)
                        {
                            string key = step.Length == 0 ? $"{start}:{stop}" : $"{start}:{stop}:{step}";
                            ExportContract.CheckAll(_failures, $"{code} a[{key}]", vector[key], value);
                        }
            }
        }

        /// <summary>Every axis-slice pair of a 3×4 matrix (and of its transpose), each also transposed after slicing.</summary>
        private sealed class MatrixSliceSweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public MatrixSliceSweep(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code != NPTypeCode.Boolean && code != NPTypeCode.Int32 && code != NPTypeCode.Decimal)
                    return;
                string[] axis = { ":", "1:", ":2", "::2", "1::2", "::-1", "::-2", "2:0:-1", "1:1", "0", "-1" };
                NDArray matrix = Layouts.Base(value, 3, 4);
                foreach (bool transposeFirst in new[] { false, true })
                    foreach (string rows in axis)
                        foreach (string cols in axis)
                        {
                            NDArray source = transposeFirst ? matrix.T : matrix;
                            string name = $"{code} {(transposeFirst ? "m.T" : "m")}[{rows}, {cols}]";
                            NDArray view = source[$"{rows}, {cols}"];
                            ExportContract.CheckAll(_failures, name, view, value);
                            ExportContract.CheckAll(_failures, name + ".T", view.T, value);
                        }
            }
        }
    }
}
