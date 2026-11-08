using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Property-based and bounded-exhaustive coverage. Seeded random NumSharp views (any-sign slices, integer indices,
    ///     permutations, expand_dims, broadcasts, read-only flags, in random combination) go through the export contract;
    ///     seeded random ORT tensors (row- and column-major, empty axes, 0-d) go through the import contract; and every slice
    ///     of a small vector and every axis-slice pair of a small matrix go through the export contract exhaustively. Seeds
    ///     are fixed, and each cell's name carries the recipe that built it, so a failure reproduces from its message alone.
    /// </summary>
    [TestClass]
    public class PropertyBasedTests : OnnxTestBase
    {
        /// <summary>The seed every generator here starts from.</summary>
        private const int Seed = 20261002;

        /// <summary>Random views per dtype.</summary>
        private const int ViewsPerDtype = 100;

        /// <summary>Random tensors per ORT element type.</summary>
        private const int TensorsPerType = 100;

        /// <summary>Random views of every dtype satisfy the export contract.</summary>
        [TestMethod]
        public void RandomViews_SatisfyTheExportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomViewCheck(failures));
            failures.AssertNone("random view x verb");
        }

        /// <summary>
        ///     Random ORT tensors of every ORT element type satisfy the import contract: <c>ToNDArray</c> and <c>AsNDArray</c>
        ///     of an <see cref="OrtValue"/> and of a <see cref="DenseTensor{T}"/> (either stride order) produce the tensor's
        ///     logical elements, dtype and shape; the zero-copy views write through to the tensor.
        /// </summary>
        [TestMethod]
        public void RandomTensors_SatisfyTheImportContract()
        {
            var failures = new Failures();
            Dtypes.ForEach(new RandomTensorCheck(failures));
            failures.AssertNone("random tensor x verb");
        }

        /// <summary>Every <c>start:stop:step</c> slice of a length-5 vector (steps ±1..±3, every bound incl. omitted) satisfies the export contract.</summary>
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
        /// <returns>The slice text, e.g. <c>1:3:-1</c> or <c>-2</c>.</returns>
        internal static string AxisSlice(Random rng, long d)
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
        ///     A random view: a C-contiguous base of random rank (1–4) and lengths (0–4), then up to three random operations
        ///     (a slice of every axis, a permutation, expand_dims, a broadcast that adds and stretches axes, a read-only flag).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rng">The random source.</param>
        /// <param name="value">The value generator.</param>
        /// <param name="recipe">Receives the recipe, so a failing case names how it was built.</param>
        /// <returns>The view.</returns>
        internal static NDArray RandomView<T>(Random rng, Func<long, T> value, StringBuilder recipe) where T : unmanaged
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

        /// <summary>Random ORT tensors through the import contract, per ORT element type.</summary>
        private sealed class RandomTensorCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal RandomTensorCheck(Failures failures) => _failures = failures;

            /// <summary>
            ///     Runs <see cref="TensorsPerType"/> random tensors of the ORT element type this dtype imports from. Char,
            ///     Decimal and Complex have no ORT element type of their own (Char's UInt16 is covered by UInt16).
            /// </summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                switch (code)
                {
                    case NPTypeCode.Char:
                    case NPTypeCode.Decimal:
                    case NPTypeCode.Complex:
                        return;
                    case NPTypeCode.Half:
                        Check<T, Float16>(code, value);
                        return;
                    default:
                        Check<T, T>(code, value);
                        return;
                }
            }

            /// <summary>The import checks for ORT element type <typeparamref name="X"/> (the bits of <typeparamref name="T"/>).</summary>
            /// <typeparam name="T">The NumSharp element type the tensor imports as.</typeparam>
            /// <typeparam name="X">The tensor's element type.</typeparam>
            /// <param name="code">The NumSharp dtype the tensor imports as.</param>
            /// <param name="value">The value generator.</param>
            private void Check<T, X>(NPTypeCode code, Func<long, T> value) where T : unmanaged where X : unmanaged
            {
                var rng = new Random(Seed ^ (int)code ^ 0x5A5A);
                for (int i = 0; i < TensorsPerType; i++)
                {
                    int rank = rng.Next(0, 5);
                    var dims = new long[rank];
                    for (int a = 0; a < rank; a++)
                        dims[a] = rng.Next(10) == 0 ? 0 : rng.Next(1, 5);
                    long size = dims.Aggregate(1L, (p, d) => p * d);
                    long offset = rng.Next(0, 50);   // distinct values per tensor
                    var source = new T[size];
                    for (long k = 0; k < size; k++)
                        source[k] = value(k + offset);
                    X[] data = MemoryMarshal.Cast<T, X>(source).ToArray();
                    byte[] bytes = Oracle.Bytes(data);
                    string cell = $"{code} #{i} {Oracle.Describe(dims)}";

                    _failures.Check($"{cell} / OrtValue.ToNDArray", () =>
                    {
                        using OrtValue v = OrtValue.CreateTensorValueFromMemory((X[])data.Clone(), dims);
                        using NDArray back = v.ToNDArray();
                        return ExportContract.Imported(back, code, dims, bytes);
                    });

                    _failures.Check($"{cell} / OrtValue.AsNDArray", () =>
                    {
                        X[] backing = (X[])data.Clone();
                        using OrtValue v = OrtValue.CreateTensorValueFromMemory(backing, dims);
                        using NDArray back = v.AsNDArray();
                        string problem = ExportContract.Imported(back, code, dims, bytes);
                        if (problem != null || size == 0)
                            return problem;
                        // Zero-copy: a write through NumSharp lands in the tensor's own memory (the managed array ORT pinned).
                        long k = rng.NextInt64(size);
                        T sentinel = Oracle.Different(back.GetAtIndex<T>(k), value);
                        back.SetAtIndex(sentinel, k);
                        return Oracle.SameBytes(MemoryMarshal.Cast<X, T>(backing)[(int)k], sentinel) ? null : "a write through the view did not reach the tensor";
                    });

                    bool reversed = rank >= 2 && rng.Next(2) == 0;
                    _failures.Check($"{cell} / DenseTensor{(reversed ? " (column-major)" : "")}.ToNDArray+AsNDArray", () =>
                    {
                        int[] intDims = Oracle.Ints(dims);
                        var tensor = new DenseTensor<X>((X[])data.Clone(), intDims, reversed);
                        // The tensor's LOGICAL elements, through its own strided indexer.
                        var logical = new X[size];
                        foreach (long[] index in Oracle.Indices(dims))
                            logical[Oracle.Flat(index, dims)] = tensor[Oracle.Ints(index)];
                        byte[] logicalBytes = Oracle.Bytes(logical);
                        using NDArray copy = tensor.ToNDArray();
                        using NDArray view = tensor.AsNDArray();
                        return ExportContract.Imported(copy, code, dims, logicalBytes)
                               ?? ExportContract.Imported(view, code, dims, logicalBytes);
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

            /// <summary>Runs every slice for the dtypes whose crossing differs (direct, reinterpreted, converted).</summary>
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
                if (code != NPTypeCode.Int32 && code != NPTypeCode.Decimal)
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
