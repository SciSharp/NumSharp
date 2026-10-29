using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     A check written once as a generic method and run for each of the 15 NumSharp element types. MSTest test
    ///     methods cannot be generic, so a test implements this and hands it to <see cref="Dtypes.ForEach"/>.
    /// </summary>
    internal interface IDtypeCheck
    {
        /// <summary>Runs the check for one element type.</summary>
        /// <typeparam name="T">The CLR element type of <paramref name="code"/>.</typeparam>
        /// <param name="code">The NumSharp dtype under test.</param>
        /// <param name="value">
        ///     The value stored at a flat index. Different small indices give different values for every type with
        ///     more than two values, so an element read from the wrong place shows up; bool can only alternate, which
        ///     is why the write-through checks also write sentinels.
        /// </param>
        void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged;
    }

    /// <summary>The 15 NumSharp element types, each paired with a value generator, for <see cref="IDtypeCheck"/> sweeps.</summary>
    internal static class Dtypes
    {
        /// <summary>
        ///     Runs <paramref name="check"/> once per NumSharp element type. Every generator gives distinct values for at
        ///     least the first 250 indices (bool excepted), which covers every array the sweeps build.
        /// </summary>
        /// <param name="check">The generic check to run.</param>
        internal static void ForEach(IDtypeCheck check)
        {
            check.Run<bool>(NPTypeCode.Boolean, i => (i * 7 + 3) % 5 < 2);
            check.Run<byte>(NPTypeCode.Byte, i => (byte)(i * 7 + 1));
            check.Run<sbyte>(NPTypeCode.SByte, i => (sbyte)(i * 37 % 251 - 125));
            check.Run<short>(NPTypeCode.Int16, i => (short)(i * 301 - 1000));
            check.Run<ushort>(NPTypeCode.UInt16, i => (ushort)(i * 301 + 5));
            check.Run<int>(NPTypeCode.Int32, i => (int)(i * 100003 - 7));
            check.Run<uint>(NPTypeCode.UInt32, i => (uint)(i * 100003 + 7));
            check.Run<long>(NPTypeCode.Int64, i => i * 10000000019L - 3);
            check.Run<ulong>(NPTypeCode.UInt64, i => (ulong)i * 10000000019UL + 3);
            check.Run<char>(NPTypeCode.Char, i => (char)(0x100 + i));
            check.Run<Half>(NPTypeCode.Half, i => (Half)(i * 0.5 - 3));
            check.Run<float>(NPTypeCode.Single, i => i * 1.25f - 7.5f);
            check.Run<double>(NPTypeCode.Double, i => i * 1.125 - 9.75);
            check.Run<decimal>(NPTypeCode.Decimal, i => i * 1.5m - 2.25m);
            check.Run<Complex>(NPTypeCode.Complex, i => new Complex(i * 1.5 - 2, 1 - i * 0.25));
        }
    }

    /// <summary>
    ///     Reference reads and index arithmetic that do not go through the bridge under test, so a test compares the
    ///     bridge against NumSharp's own view of the data (or the BCL's own view of a tensor).
    /// </summary>
    internal static class Oracle
    {
        /// <summary>
        ///     The array's elements in logical C order, read one at a time through NumSharp's own indexer
        ///     (<see cref="NDArray.GetAtIndex{T}"/> maps a flat logical index through the view's strides and offset).
        /// </summary>
        /// <typeparam name="T">The array's element type.</typeparam>
        /// <param name="nd">The array (any layout).</param>
        /// <returns>A new array of <c>nd.size</c> elements.</returns>
        internal static T[] Elements<T>(NDArray nd) where T : unmanaged
        {
            var result = new T[nd.size];
            for (long i = 0; i < result.LongLength; i++)
                result[i] = nd.GetAtIndex<T>(i);
            return result;
        }

        /// <summary>A span's elements in its logical order, through the BCL's own <c>FlattenTo</c>.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="span">The span to read.</param>
        /// <returns>A new array of <c>span.FlattenedLength</c> elements.</returns>
        internal static T[] Flatten<T>(in ReadOnlyTensorSpan<T> span)
        {
            var result = new T[(long)span.FlattenedLength];
            span.FlattenTo(result);
            return result;
        }

        /// <summary>A tensor's elements in its logical order, through the BCL's own <c>FlattenTo</c>.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="tensor">The tensor to read.</param>
        /// <returns>A new array of <c>tensor.FlattenedLength</c> elements.</returns>
        internal static T[] Flatten<T>(Tensor<T> tensor)
        {
            var result = new T[(long)tensor.FlattenedLength];
            tensor.FlattenTo(result);
            return result;
        }

        /// <summary>Byte equality of two arrays: NaN payloads, signed zeros, bool bytes and decimal scales all count.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The first array.</param>
        /// <param name="b">The second array.</param>
        /// <returns><c>true</c> when both hold the same bytes.</returns>
        internal static bool SameBytes<T>(T[] a, T[] b) where T : unmanaged
            => MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan()));

        /// <summary>Byte equality of two values (see <see cref="SameBytes{T}(T[], T[])"/>).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns><c>true</c> when both hold the same bytes.</returns>
        internal static bool SameBytes<T>(T a, T b) where T : unmanaged
            => SameBytes(new[] { a }, new[] { b });

        /// <summary>A value whose bytes differ from <paramref name="old"/>: the sentinel a write-through check writes.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="old">The value being replaced.</param>
        /// <param name="value">The type's value generator (indices from 1000 on are past every array's own values).</param>
        /// <returns>The first generated value, from index 1000 on, that differs from <paramref name="old"/>.</returns>
        internal static T Different<T>(T old, Func<long, T> value) where T : unmanaged
        {
            for (long k = 1000; ; k++)
            {
                T candidate = value(k);
                if (!SameBytes(candidate, old))
                    return candidate;
            }
        }

        /// <summary>Every multi-index of a shape in C order: none for an empty shape, one empty index for a 0-d shape.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The multi-indices, each a fresh array.</returns>
        internal static IEnumerable<long[]> Indices(long[] dims)
        {
            foreach (long d in dims)
                if (d == 0)
                    yield break;
            var index = new long[dims.Length];
            while (true)
            {
                yield return (long[])index.Clone();
                int axis = dims.Length - 1;
                while (axis >= 0 && ++index[axis] == dims[axis])
                    index[axis--] = 0;
                if (axis < 0)
                    yield break;
            }
        }

        /// <summary>The C-order flat position of a multi-index.</summary>
        /// <param name="index">The multi-index.</param>
        /// <param name="dims">The shape it indexes.</param>
        /// <returns>The flat position.</returns>
        internal static long Flat(long[] index, long[] dims)
        {
            long flat = 0;
            for (int i = 0; i < dims.Length; i++)
                flat = flat * dims[i] + index[i];
            return flat;
        }

        /// <summary>The C-order multi-index of a flat position.</summary>
        /// <param name="flat">The flat position.</param>
        /// <param name="dims">The shape.</param>
        /// <returns>The multi-index.</returns>
        internal static long[] Unravel(long flat, long[] dims)
        {
            var index = new long[dims.Length];
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                index[i] = flat % dims[i];
                flat /= dims[i];
            }
            return index;
        }

        /// <summary>A multi-index as the <c>nint[]</c> the BCL's indexers take.</summary>
        /// <param name="index">The multi-index.</param>
        /// <returns>The same index as <c>nint</c>s.</returns>
        internal static nint[] NInts(long[] index) => index.Select(i => (nint)i).ToArray();

        /// <summary>BCL lengths as <c>long</c>s, comparable with <see cref="NDArray.shape"/>.</summary>
        /// <param name="lengths">The lengths.</param>
        /// <returns>The same lengths as <c>long</c>s.</returns>
        internal static long[] Longs(ReadOnlySpan<nint> lengths)
        {
            var result = new long[lengths.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = lengths[i];
            return result;
        }

        /// <summary>A shape as text, e.g. <c>(3,0,4)</c>; ranks above 6 are summarized.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The description.</returns>
        internal static string Describe(long[] dims)
            => dims.Length > 6 ? $"rank-{dims.Length}" : "(" + string.Join(",", dims) + ")";

        /// <summary>
        ///     Does the view step through memory backwards? Only an axis longer than 1 counts (a length-1 axis is never
        ///     stepped), and an empty view addresses nothing — the exact condition under which <c>AsTensorSpan</c> must
        ///     refuse, since <c>System.Numerics.Tensors</c> forbids negative strides.
        /// </summary>
        /// <param name="nd">The view.</param>
        /// <returns><c>true</c> when the view has a negative stride on an axis it steps along.</returns>
        internal static bool HasNegativeStride(NDArray nd)
        {
            if (nd.size == 0)
                return false;
            long[] dims = nd.shape;
            long[] strides = nd.Shape.strides;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] > 1 && strides[i] < 0)
                    return true;
            return false;
        }

        /// <summary>Does the view repeat memory through a zero stride on an axis longer than 1 (a broadcast)?</summary>
        /// <param name="dims">The lengths.</param>
        /// <param name="strides">The element strides.</param>
        /// <returns><c>true</c> when two different indices address the same element.</returns>
        internal static bool HasZeroStride(long[] dims, long[] strides)
        {
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] > 1 && strides[i] == 0)
                    return true;
            return false;
        }
    }

    /// <summary>
    ///     Collects the failing cells of a sweep, so one assertion reports all of them instead of the first one hiding
    ///     every later failure.
    /// </summary>
    internal sealed class Failures
    {
        private readonly List<string> _problems = new List<string>();

        /// <summary>The number of cells checked so far.</summary>
        internal int Cells { get; private set; }

        /// <summary>
        ///     Runs one cell. The check returns <c>null</c> when the cell holds and a description of the problem when it
        ///     does not; an exception it throws is recorded as the problem.
        /// </summary>
        /// <param name="cell">The cell's name (dtype, layout, verb, seed — whatever reproduces it).</param>
        /// <param name="check">The check.</param>
        internal void Check(string cell, Func<string> check)
        {
            Cells++;
            try
            {
                string problem = check();
                if (problem != null)
                    _problems.Add($"{cell}: {problem}");
            }
            catch (Exception e)
            {
                _problems.Add($"{cell}: threw {e.GetType().Name}: {e.Message.Replace('\n', ' ')}");
            }
        }

        /// <summary>Records a problem found outside <see cref="Check"/> (for example while building a case).</summary>
        /// <param name="cell">The cell's name.</param>
        /// <param name="problem">The problem.</param>
        internal void Add(string cell, string problem)
        {
            Cells++;
            _problems.Add($"{cell}: {problem}");
        }

        /// <summary>Fails the test, listing up to 60 failing cells, unless every cell held.</summary>
        /// <param name="what">What a cell is, for the message (e.g. "dtype x layout").</param>
        /// <exception cref="AssertFailedException">At least one cell failed, or no cell ran at all.</exception>
        internal void AssertNone(string what)
        {
            Assert.IsTrue(Cells > 0, $"no {what} cell ran");
            if (_problems.Count == 0)
            {
                // Logged on success too, so a sweep that silently shrank (a generator or catalog change) shows in the logs.
                Console.WriteLine($"{Cells} {what} cells checked, all held");
                return;
            }
            string shown = string.Join("\n  ", _problems.Take(60));
            string more = _problems.Count > 60 ? $"\n  … and {_problems.Count - 60} more" : "";
            Assert.Fail($"{_problems.Count} of {Cells} {what} cells failed:\n  {shown}{more}");
        }
    }

    /// <summary>The NumSharp array layouts the matrix and property tests run the bridge over.</summary>
    internal static class Layouts
    {
        /// <summary>
        ///     A C-contiguous array of the given shape holding <paramref name="value"/>(i) at flat index i. A 1-D shape
        ///     is the array itself; any other shape is a reshaped view of it (the 1-D owner stays alive through the view).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The value generator.</param>
        /// <param name="dims">The shape (at least one dimension).</param>
        /// <returns>The array.</returns>
        internal static NDArray Base<T>(Func<long, T> value, params long[] dims) where T : unmanaged
        {
            long n = 1;
            foreach (long d in dims)
                n *= d;
            var values = new T[n];
            for (long i = 0; i < n; i++)
                values[i] = value(i);
            NDArray flat = np.array(values);
            return dims.Length == 1 ? flat : flat.reshape(new Shape(dims));
        }

        /// <summary>A rank-<paramref name="rank"/> shape of mostly unit axes with three longer ones (2, 3, 2), 12 elements.</summary>
        /// <param name="rank">The rank (at least 3).</param>
        /// <returns>The shape.</returns>
        internal static long[] HighRank(int rank)
        {
            var dims = Enumerable.Repeat(1L, rank).ToArray();
            dims[0] = 2;
            dims[rank / 2] = 3;
            dims[rank - 1] = 2;
            return dims;
        }

        /// <summary>
        ///     Every named layout, each built fresh by its factory: contiguous at several ranks, 0-d, empty (also at an
        ///     offset and reversed), offset / row / column / single-row / single-column / stepped / wide-step slices,
        ///     transposes and permutations, Fortran order, unit axes, broadcasts (row, column, 0-d, sliced, made
        ///     writeable), a diagonal, a read-only flag, negative strides (refused) and a negative stride on a unit axis
        ///     (accepted), and ranks 40 and 70.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The value generator.</param>
        /// <returns>The layouts, by name.</returns>
        internal static List<(string Name, Func<NDArray> Build)> All<T>(Func<long, T> value) where T : unmanaged
        {
            NDArray B(params long[] d) => Base(value, d);
            return new List<(string, Func<NDArray>)>
            {
                ("C 1-D (6)", () => B(6)),
                ("C 2-D (2,3)", () => B(2, 3)),
                ("C 3-D (2,3,4)", () => B(2, 3, 4)),
                ("C 4-D (2,1,3,2)", () => B(2, 1, 3, 2)),
                ("0-d scalar", () => NDArray.Scalar(value(4))),
                ("empty (0,)", () => B(0)),
                ("empty (3,0)", () => B(3, 0)),
                ("empty (0,3)", () => B(0, 3)),
                ("empty (2,0,4)", () => B(2, 0, 4)),
                ("empty slice at an offset a[5:5]", () => B(8)["5:5"]),
                ("empty column slice m[:, 3:3]", () => B(4, 5)[":, 3:3"]),
                ("empty reversed slice a[5:5:-1]", () => B(8)["5:5:-1"]),
                ("last element a[-1:]", () => B(8)["-1:"]),
                ("offset slice a[2:5]", () => B(8)["2:5"]),
                ("row slice m[1:3]", () => B(4, 5)["1:3"]),
                ("column slice m[:, 1:3]", () => B(4, 5)[":, 1:3"]),
                ("single row m[2]", () => B(4, 5)["2"]),
                ("single column m[:, 2]", () => B(4, 5)[":, 2"]),
                ("stepped a[::2]", () => B(10)["::2"]),
                ("wide step a[::7]", () => B(50)["::7"]),
                ("stepped 2-D m[::2, 1::3]", () => B(6, 7)["::2, 1::3"]),
                ("transposed m.T", () => B(3, 4).T),
                ("permuted 3-D (2,0,1)", () => B(2, 3, 4).transpose(new[] { 2, 0, 1 })),
                ("swapaxes 3-D (0,2)", () => np.swapaxes(B(2, 3, 4), 0, 2)),
                ("F-contiguous", () => np.asfortranarray(B(3, 4))),
                ("transposed slice m.T[1:3, ::2]", () => B(4, 5).T["1:3, ::2"]),
                ("expand_dims (1,6)", () => np.expand_dims(B(6), 0)),
                ("expand_dims (2,1,3)", () => np.expand_dims(B(2, 3), 1)),
                ("unit axes (3,1)", () => B(3, 1)),
                ("unit axes (1,4,1)", () => B(1, 4, 1)),
                ("broadcast row -> (3,4)", () => np.broadcast_to(B(4), new Shape(3, 4))),
                ("broadcast column -> (3,4)", () => np.broadcast_to(B(3, 1), new Shape(3, 4))),
                ("broadcast 0-d -> (2,3)", () => np.broadcast_to(NDArray.Scalar(value(5)), new Shape(2, 3))),
                ("sliced broadcast", () => np.broadcast_to(B(4), new Shape(3, 4))["1:3, ::2"]),
                ("broadcast made writeable", () =>
                {
                    NDArray b = np.broadcast_to(B(4), new Shape(3, 4));
                    b.setflags(write: true);
                    return b;
                }),
                ("diagonal view", () => np.diagonal(B(4, 4))),
                ("read-only flag", () =>
                {
                    NDArray a = B(2, 3);
                    a.setflags(write: false);
                    return a;
                }),
                ("reversed a[::-1]", () => B(5)["::-1"]),
                ("reversed columns m[:, ::-1]", () => B(3, 4)[":, ::-1"]),
                ("reversed rows m[::-1]", () => B(3, 4)["::-1"]),
                ("reversed stepped a[::-2]", () => B(9)["::-2"]),
                ("reversed unit axis (1,5)[::-1]", () => B(1, 5)["::-1"]),
                ("reversed unit axis (2,1,3)[:, ::-1]", () => B(2, 1, 3)[":, ::-1"]),
                ("rank 40", () => B(HighRank(40))),
                ("rank 70", () => B(HighRank(70))),
            };
        }
    }

    /// <summary>
    ///     The bridge's export contract for ONE NumSharp view, checked against NumSharp's own reads: the matrix test
    ///     runs it over every catalogued layout, the property tests over random views. Every verb is checked, and a
    ///     view the contract says must be refused (a negative stride on an axis it steps) is checked for the refusal.
    /// </summary>
    internal static class ExportContract
    {
        /// <summary>The most elements whose writes are checked one by one per view; larger views write a sample.</summary>
        private const int MaxIndividualWrites = 64;

        /// <summary>Runs every export check over <paramref name="view"/>, recording each verb as its own cell.</summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name (dtype, layout or random-case description).</param>
        /// <param name="view">The view to export.</param>
        /// <param name="value">The type's value generator (for write-through sentinels).</param>
        /// <param name="rng">When set, larger views write a random sample of indices instead of the first ones.</param>
        internal static void CheckAll<T>(Failures failures, string cell, NDArray view, Func<long, T> value, Random rng = null) where T : unmanaged
        {
            T[] want = Oracle.Elements<T>(view);
            long[] dims = view.shape;
            // The BCL has no rank 0: a 0-d array crosses as the single-element vector [1].
            long[] crossed = dims.Length == 0 ? new long[] { 1 } : dims;
            bool refuse = Oracle.HasNegativeStride(view);
            NPTypeCode code = view.typecode;

            failures.Check($"{cell} / AsTensorSpan", () =>
            {
                if (refuse)
                {
                    try
                    {
                        using var refused = view.AsTensorSpan<T>();
                        return "accepted a negative-stride view (System.Numerics.Tensors forbids negative strides)";
                    }
                    catch (InvalidOperationException e) when (e.Message.Contains("negative-stride"))
                    {
                        return null;
                    }
                }

                using var h = view.AsTensorSpan<T>();
                ReadOnlyTensorSpan<T> ro = h.ReadOnlySpan;
                if (!Oracle.Longs(ro.Lengths).SequenceEqual(crossed))
                    return $"span lengths {Oracle.Describe(Oracle.Longs(ro.Lengths))}, expected {Oracle.Describe(crossed)}";
                if (!h.Lengths.Select(x => (long)x).SequenceEqual(crossed))
                    return "handle.Lengths disagree with the span's";
                if (!Oracle.SameBytes(Oracle.Flatten(ro), want))
                    return "FlattenTo bytes differ from NumSharp's elements";
                // The BCL indexer applies the strides itself (a different path than FlattenTo): read EVERY element.
                foreach (long[] index in Oracle.Indices(crossed))
                    if (!Oracle.SameBytes(ro[Oracle.NInts(index)], want[Oracle.Flat(index, crossed)]))
                        return $"span[{string.Join(",", index)}] differs from NumSharp's element";
                if (h.IsWriteable != view.Shape.IsWriteable)
                    return $"handle.IsWriteable {h.IsWriteable}, the view's flag is {view.Shape.IsWriteable}";
                return null;
            });

            if (!refuse)
                failures.Check($"{cell} / write-through", () => CheckWrites(view, want, crossed, value, rng));

            failures.Check($"{cell} / ToTensor", () =>
            {
                Tensor<T> t = view.ToTensor<T>();
                if (!Oracle.Longs(t.Lengths).SequenceEqual(crossed))
                    return $"tensor lengths {Oracle.Describe(Oracle.Longs(t.Lengths))}, expected {Oracle.Describe(crossed)}";
                return Oracle.SameBytes(Oracle.Flatten(t), want) ? null : "tensor bytes differ from NumSharp's elements";
            });

            failures.Check($"{cell} / ToTensor->ToNDArray", () =>
            {
                using NDArray back = view.ToTensor<T>().ToNDArray();
                return Imported(back, code, crossed, want);
            });

            failures.Check($"{cell} / ToTensor->AsNDArray", () =>
            {
                using NDArray back = view.ToTensor<T>().AsNDArray();
                return Imported(back, code, crossed, want);
            });

            if (!refuse)
                failures.Check($"{cell} / AsTensorSpan->span.ToNDArray", () =>
                {
                    using var h = view.AsTensorSpan<T>();
                    using NDArray back = h.ReadOnlySpan.ToNDArray();
                    return Imported(back, code, crossed, want);
                });
        }

        /// <summary>
        ///     Writes through the mutable span and checks NumSharp sees exactly that write. For a view whose indices
        ///     address distinct elements, every index (or a sample) is written in turn and ONLY the element it names may
        ///     change — a stride bug that maps two indices onto one element, or an index onto the wrong element, shows
        ///     up. A broadcast made writeable aliases elements by design, so it gets one write and one read back.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="view">The view.</param>
        /// <param name="want">NumSharp's elements before any write.</param>
        /// <param name="crossed">The span's lengths (the view's shape, or [1] for 0-d).</param>
        /// <param name="value">The value generator, for sentinels.</param>
        /// <param name="rng">When set, a view with more than the cap writes a random sample of indices.</param>
        /// <returns><c>null</c> when every write landed where it should, else the problem.</returns>
        private static string CheckWrites<T>(NDArray view, T[] want, long[] crossed, Func<long, T> value, Random rng) where T : unmanaged
        {
            using var h = view.AsTensorSpan<T>();
            if (!h.IsWriteable)
            {
                try
                {
                    _ = h.Span;
                    return "Span was handed out for a read-only view";
                }
                catch (InvalidOperationException)
                {
                    return null;
                }
            }
            if (want.Length == 0)
            {
                h.Span.Fill(default);   // an empty writeable span must still be usable
                return null;
            }

            TensorSpan<T> span = h.Span;
            bool aliased = Oracle.HasZeroStride(view.shape, view.Shape.strides);
            IEnumerable<long[]> targets = Oracle.Indices(crossed);
            if (aliased)
                targets = targets.Take(1);
            else if (want.Length > MaxIndividualWrites)
                targets = rng == null
                    ? targets.Take(MaxIndividualWrites)
                    : Enumerable.Range(0, 16).Select(_ => Oracle.Unravel(rng.NextInt64(want.Length), crossed));

            foreach (long[] index in targets)
            {
                long flat = Oracle.Flat(index, crossed);
                nint[] at = Oracle.NInts(index);
                T old = span[at];
                T sentinel = Oracle.Different(old, value);
                span[at] = sentinel;
                T[] now = Oracle.Elements<T>(view);
                span[at] = old;

                if (!Oracle.SameBytes(now[flat], sentinel))
                    return $"a write through span[{string.Join(",", index)}] is not visible at that NumSharp element";
                if (aliased)
                    continue;
                for (long i = 0; i < now.LongLength; i++)
                    if (i != flat && !Oracle.SameBytes(now[i], want[i]))
                        return $"a write through span[{string.Join(",", index)}] also changed NumSharp element {i}";
            }
            return Oracle.SameBytes(Oracle.Elements<T>(view), want) ? null : "the restoring writes did not restore the view";
        }

        /// <summary>Checks an imported array's shape, dtype and elements.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="back">The imported array.</param>
        /// <param name="code">The expected dtype.</param>
        /// <param name="crossed">The expected shape.</param>
        /// <param name="want">The expected elements.</param>
        /// <returns><c>null</c> when all three match, else the problem.</returns>
        internal static string Imported<T>(NDArray back, NPTypeCode code, long[] crossed, T[] want) where T : unmanaged
        {
            if (back.typecode != code)
                return $"dtype {back.typecode}, expected {code}";
            if (!back.shape.SequenceEqual(crossed))
                return $"shape {Oracle.Describe(back.shape)}, expected {Oracle.Describe(crossed)}";
            return Oracle.SameBytes(Oracle.Elements<T>(back), want) ? null : "elements differ";
        }
    }
}
