using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Property-based and bounded-exhaustive coverage. Seeded random NumSharp views (any-sign slices, integer indices,
    ///     transposes, expand_dims, broadcasts, read-only flags — biased to the ranks the data views take) go through the
    ///     export contract; seeded random <see cref="VBuffer{T}"/>s (dense and sparse, backing arrays longer than the vector,
    ///     empty) go through the import contract; and every slice of a small vector and every axis-slice pair of a small
    ///     matrix go through the export contract exhaustively. Seeds are fixed and each cell names its recipe.
    /// </summary>
    [TestClass]
    public class PropertyBasedTests : MLNetTestBase
    {
        /// <summary>The seed every generator here starts from.</summary>
        private const int Seed = 20261002;

        /// <summary>Random views per dtype.</summary>
        private const int ViewsPerDtype = 80;

        /// <summary>Random buffers per ML.NET item type.</summary>
        private const int BuffersPerType = 150;

        /// <summary>Random views of every dtype satisfy the export contract.</summary>
        [TestMethod]
        public void RandomViews_SatisfyTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomViewCheck(failures));
            failures.AssertNone("random view x verb");
        }

        /// <summary>
        ///     Random <see cref="VBuffer{T}"/>s satisfy the import contract: <c>ToNDArray</c> densifies (a missing entry is
        ///     zero), <c>AsNDArray</c> reads the same values, shares a dense buffer's own array (a write through the view is
        ///     visible in the buffer) and copies a sparse one.
        /// </summary>
        [TestMethod]
        public void RandomVBuffers_SatisfyTheImportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomBufferCheck(failures));
            failures.AssertNone("random VBuffer x verb");
        }

        /// <summary>Every <c>start:stop:step</c> slice of a length-5 vector satisfies the export contract.</summary>
        [TestMethod]
        public void EverySliceOfAVector_SatisfiesTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new VectorSliceSweep(failures));
            failures.AssertNone("vector slice x verb");
        }

        /// <summary>Every pair of axis slices of a 3×4 matrix and of its transpose satisfies the export contract.</summary>
        [TestMethod]
        public void EveryAxisSlicePairOfAMatrix_SatisfiesTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new MatrixSliceSweep(failures));
            failures.AssertNone("matrix slice x verb");
        }

        // ================================================================================== generators

        /// <summary>A random slice of one axis of length <paramref name="d"/>: any step sign, any bounds, sometimes an integer index.</summary>
        /// <param name="rng">The random source.</param>
        /// <param name="d">The axis length.</param>
        /// <returns>The slice text.</returns>
        private static string AxisSlice(Random rng, long d)
        {
            if (d > 0 && rng.Next(8) == 0)
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
                long a = d == 0 ? -1 : rng.NextInt64(0, d);
                long b = rng.NextInt64(-1, a + 1);
                start = d == 0 || rng.Next(3) == 0 ? "" : a.ToString();
                stop = b < 0 || rng.Next(3) == 0 ? "" : b.ToString();
            }
            return $"{start}:{stop}:{step}";
        }

        /// <summary>
        ///     A random view biased to the data views' ranks: a base of rank 1–2 (sometimes 3), lengths 0–6, then up to three
        ///     random operations (slices, a transpose, expand_dims, a broadcast, a read-only flag).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rng">The random source.</param>
        /// <param name="value">The value generator.</param>
        /// <param name="recipe">Receives the recipe.</param>
        /// <returns>The view.</returns>
        private static NDArray RandomView<T>(Random rng, Func<long, T> value, StringBuilder recipe) where T : unmanaged
        {
            int rank = rng.Next(10) == 0 ? 3 : rng.Next(1, 3);
            var dims = new long[rank];
            for (int i = 0; i < rank; i++)
                dims[i] = rng.Next(12) == 0 ? 0 : rng.Next(1, 7);
            NDArray view = Layouts.Base(value, dims);
            recipe.Append("base").Append(Oracle.Describe(dims));
            int operations = rng.Next(0, 4);
            for (int k = 0; k < operations; k++)
            {
                switch (rng.Next(5))
                {
                    case 0:
                        string key = string.Join(", ", view.shape.Select(d => AxisSlice(rng, d)));
                        if (view.ndim == 0)
                            break;
                        view = view[key];
                        recipe.Append($" [{key}]");
                        break;
                    case 1:
                        if (view.ndim < 2)
                            break;
                        view = view.T;
                        recipe.Append(" .T");
                        break;
                    case 2:
                        if (view.ndim > 1)
                            break;
                        int axis = rng.Next(0, view.ndim + 1);
                        view = np.expand_dims(view, axis);
                        recipe.Append($" expand_dims({axis})");
                        break;
                    case 3:
                        if (view.ndim > 1)
                            break;
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

        // ================================================================================== checks

        /// <summary>Random views of one dtype through the export contract.</summary>
        private sealed class RandomViewCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal RandomViewCheck(Failures failures) => _failures = failures;

            /// <summary>Runs <see cref="ViewsPerDtype"/> random views of one dtype.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var rng = new Random(Seed ^ (int)code);
                for (int i = 0; i < ViewsPerDtype; i++)
                {
                    var recipe = new StringBuilder();
                    NDArray view = RandomView(rng, value, recipe);
                    ExportContract.CheckAll(_failures, $"{code} #{i} {recipe}", view, value);
                }
            }
        }

        /// <summary>Random VBuffers through the import contract, per ML.NET item type.</summary>
        private sealed class RandomBufferCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal RandomBufferCheck(Failures failures) => _failures = failures;

            /// <summary>Runs <see cref="BuffersPerType"/> random buffers of the ML.NET item type this dtype imports from.</summary>
            /// <typeparam name="T">The element type (Char, Half, Decimal and Complex have no ML.NET item type of their own).</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code == NPTypeCode.Char || code == NPTypeCode.Half || code == NPTypeCode.Decimal || code == NPTypeCode.Complex)
                    return;
                var rng = new Random(Seed ^ (int)code ^ 0x3C3C);
                for (int i = 0; i < BuffersPerType; i++)
                {
                    int length = rng.Next(8) == 0 ? 0 : rng.Next(1, 40);
                    bool sparse = rng.Next(2) == 0 && length > 0;
                    int count = sparse ? rng.Next(0, length + 1) : length;
                    int slack = rng.Next(0, 4);   // backing arrays longer than needed, as pooled ML.NET buffers are
                    var values = new T[count + slack];
                    for (int k = 0; k < values.Length; k++)
                        values[k] = value(rng.Next(0, 200));
                    int[] indices = null;
                    var dense = new T[length];
                    if (sparse)
                    {
                        indices = Enumerable.Range(0, length).OrderBy(_ => rng.Next()).Take(count).OrderBy(x => x).ToArray();
                        indices = indices.Concat(Enumerable.Repeat(0, slack)).ToArray();
                        for (int k = 0; k < count; k++)
                            dense[indices[k]] = values[k];
                    }
                    else
                    {
                        Array.Copy(values, dense, length);
                    }
                    var buffer = sparse ? new VBuffer<T>(length, count, values, indices) : new VBuffer<T>(length, values);
                    byte[] want = Oracle.Bytes(dense);
                    string cell = $"{code} #{i} length {length} {(sparse ? $"sparse {count}" : "dense")} slack {slack}";

                    _failures.Check($"{cell} / ToNDArray", () =>
                    {
                        using NDArray back = buffer.ToNDArray();
                        return ExportContract.Imported(back, code, new long[] { length }, want);
                    });

                    _failures.Check($"{cell} / AsNDArray", () =>
                    {
                        using NDArray back = buffer.AsNDArray();
                        string problem = ExportContract.Imported(back, code, new long[] { length }, want);
                        if (problem != null || sparse || length == 0)
                            return problem;
                        // Zero-copy for a dense buffer: a write through the view lands in the buffer's own values.
                        int k = rng.Next(length);
                        T sentinel = Oracle.Different(back.GetAtIndex<T>(k), value);
                        back.SetAtIndex(sentinel, k);
                        return Oracle.SameBytes(buffer.GetValues()[k], sentinel) ? null : "a write through the view did not reach the buffer";
                    });
                }
            }
        }

        /// <summary>Every slice of a length-5 vector through the export contract.</summary>
        private sealed class VectorSliceSweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal VectorSliceSweep(Failures failures) => _failures = failures;

            /// <summary>Runs every slice for a direct, a reinterpreted and two converted dtypes.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code != NPTypeCode.Single && code != NPTypeCode.Half && code != NPTypeCode.Decimal && code != NPTypeCode.Char)
                    return;
                string[] bounds = { "", "-6", "-5", "-1", "0", "1", "2", "4", "5", "6" };
                foreach (string start in bounds)
                foreach (string stop in bounds)
                foreach (int step in new[] { 1, 2, 3, -1, -2, -3 })
                {
                    string key = $"{start}:{stop}:{step}";
                    NDArray view = Layouts.Base(value, 5)[key];
                    ExportContract.CheckAll(_failures, $"{code} [{key}]", view, value);
                }
            }
        }

        /// <summary>Every axis-slice pair of a 3×4 matrix and its transpose through the export contract.</summary>
        private sealed class MatrixSliceSweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal MatrixSliceSweep(Failures failures) => _failures = failures;

            /// <summary>Runs every slice pair for a direct and a converted dtype.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code != NPTypeCode.Int32 && code != NPTypeCode.Half)
                    return;
                string[] axis = { ":", "1:", ":-1", "::2", "1::2", "::-1", "-2::-1", "1:2", "2:2", "1" };
                foreach (bool transposed in new[] { false, true })
                foreach (string a in axis)
                foreach (string b in axis)
                {
                    NDArray m = Layouts.Base(value, 3, 4);
                    if (transposed)
                        m = m.T;
                    NDArray view = m[$"{a}, {b}"];
                    ExportContract.CheckAll(_failures, $"{code} {(transposed ? "m.T" : "m")}[{a}, {b}]", view, value);
                }
            }
        }
    }
}
