using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     A check written once as a generic method and run for each of the 15 NumSharp element types. MSTest test methods
    ///     cannot be generic, so a test implements this and hands it to <see cref="Dtypes.ForEach"/>.
    /// </summary>
    internal interface IDtypeCheck
    {
        /// <summary>Runs the check for one element type.</summary>
        /// <typeparam name="T">The CLR element type of <paramref name="code"/>.</typeparam>
        /// <param name="code">The NumSharp dtype under test.</param>
        /// <param name="value">The value stored at a flat index (distinct for small indices, bool excepted).</param>
        void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged;
    }

    /// <summary>The 15 NumSharp element types, each paired with a value generator, for <see cref="IDtypeCheck"/> sweeps.</summary>
    internal static class Dtypes
    {
        /// <summary>
        ///     Runs <paramref name="check"/> once per NumSharp element type. The Half values are exact in float32 and the
        ///     Decimal values exact in float64, so the conversions the bridge performs are checkable value for value.
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
    ///     Reference reads that do not go through the bridge under test: NumSharp's own element reads, and ML.NET's own
    ///     cursor machinery reading a data view.
    /// </summary>
    internal static class Oracle
    {
        /// <summary>The array's elements in logical C order, read one at a time through NumSharp's own indexer.</summary>
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

        /// <summary>The array's element bytes in logical C order, whatever its dtype (a NumSharp C-order copy, read raw).</summary>
        /// <param name="nd">The array (any layout).</param>
        /// <returns>The bytes; empty for an empty array.</returns>
        internal static unsafe byte[] LogicalBytes(NDArray nd)
        {
            using NDArray dense = nd.copy();
            long nbytes = dense.size * dense.dtypesize;
            var bytes = new byte[nbytes];
            if (nbytes > 0)
                new ReadOnlySpan<byte>((byte*)dense.Storage.InternalArray.Address + dense.Shape.Offset * dense.dtypesize, (int)nbytes).CopyTo(bytes);
            return bytes;
        }

        /// <summary>The raw bytes of an array of unmanaged values.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="values">The values.</param>
        /// <returns>A new byte array.</returns>
        internal static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

        /// <summary>Byte equality: NaN payloads, signed zeros and bool bytes all count.</summary>
        /// <param name="a">The first byte sequence.</param>
        /// <param name="b">The second byte sequence.</param>
        /// <returns><c>true</c> when both hold the same bytes.</returns>
        internal static bool SameBytes(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) => a.SequenceEqual(b);

        /// <summary>Byte equality of two values.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns><c>true</c> when both hold the same bytes.</returns>
        internal static bool SameBytes<T>(T a, T b) where T : unmanaged
            => MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in a)).SequenceEqual(MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in b)));

        /// <summary>A value whose bytes differ from <paramref name="old"/>: the sentinel a write check writes.</summary>
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

        /// <summary>A shape as text, e.g. <c>(3,0,4)</c>; ranks above 6 are summarized.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The description.</returns>
        internal static string Describe(long[] dims)
            => dims.Length > 6 ? $"rank-{dims.Length}" : "(" + string.Join(",", dims) + ")";

        /// <summary>
        ///     The dtype an array of <paramref name="code"/> reads back as after crossing to ML.NET and back: Half is converted
        ///     to Single, Decimal to Double, Char crosses as (and comes back as) UInt16.
        /// </summary>
        /// <param name="code">The source dtype.</param>
        /// <returns>The imported dtype.</returns>
        internal static NPTypeCode CrossedCode(NPTypeCode code) => code switch
        {
            NPTypeCode.Char => NPTypeCode.UInt16,
            NPTypeCode.Half => NPTypeCode.Single,
            NPTypeCode.Decimal => NPTypeCode.Double,
            _ => code,
        };

        /// <summary>
        ///     NumSharp's elements as the values that cross, computed here without NumSharp: Char's bits as UInt16, a Half
        ///     widened to float32 (exact for every Half), a Decimal through <see cref="decimal.ToDouble"/>, every other dtype
        ///     unchanged.
        /// </summary>
        /// <typeparam name="T">The source element type.</typeparam>
        /// <typeparam name="X">The crossed element type.</typeparam>
        /// <param name="want">The source elements.</param>
        /// <returns>The crossed values in the same order.</returns>
        internal static X[] Cross<T, X>(T[] want) where T : unmanaged where X : unmanaged
        {
            if (typeof(T) == typeof(decimal))
                return (X[])(object)Array.ConvertAll((decimal[])(object)want, decimal.ToDouble);
            if (typeof(T) == typeof(Half))
                return (X[])(object)Array.ConvertAll((Half[])(object)want, h => (float)h);
            return MemoryMarshal.Cast<T, X>(want).ToArray();
        }

        /// <summary>
        ///     Reads every row of a vector column through ML.NET's own cursor machinery (a cursor, a typed getter, the
        ///     row's dense values), the way any ML.NET consumer reads it.
        /// </summary>
        /// <typeparam name="X">The column's item type.</typeparam>
        /// <param name="view">The data view.</param>
        /// <param name="column">The column's name.</param>
        /// <returns>One array per row.</returns>
        internal static List<X[]> VectorRows<X>(IDataView view, string column)
        {
            DataViewSchema.Column col = view.Schema[column];
            using DataViewRowCursor cursor = view.GetRowCursor(new[] { col });
            ValueGetter<VBuffer<X>> getter = cursor.GetGetter<VBuffer<X>>(col);
            var rows = new List<X[]>();
            var buffer = default(VBuffer<X>);
            while (cursor.MoveNext())
            {
                getter(ref buffer);
                rows.Add(buffer.DenseValues().ToArray());
            }
            return rows;
        }

        /// <summary>Reads every row of the given scalar columns through ML.NET's own cursor machinery.</summary>
        /// <typeparam name="X">The columns' type.</typeparam>
        /// <param name="view">The data view.</param>
        /// <param name="columns">The columns' names.</param>
        /// <returns>One array per row, holding the columns' values in the given order.</returns>
        internal static List<X[]> ScalarRows<X>(IDataView view, IReadOnlyList<string> columns)
        {
            DataViewSchema.Column[] cols = columns.Select(c => view.Schema[c]).ToArray();
            using DataViewRowCursor cursor = view.GetRowCursor(cols);
            ValueGetter<X>[] getters = cols.Select(c => cursor.GetGetter<X>(c)).ToArray();
            var rows = new List<X[]>();
            while (cursor.MoveNext())
            {
                var row = new X[getters.Length];
                for (int j = 0; j < getters.Length; j++)
                    getters[j](ref row[j]);
                rows.Add(row);
            }
            return rows;
        }
    }

    /// <summary>
    ///     Collects the failing cells of a sweep, so one assertion reports all of them instead of the first one hiding every
    ///     later failure.
    /// </summary>
    internal sealed class Failures
    {
        private readonly List<string> _problems = new List<string>();

        /// <summary>The number of cells checked so far.</summary>
        internal int Cells { get; private set; }

        /// <summary>
        ///     Runs one cell. The check returns <c>null</c> when the cell holds and a description of the problem when it does
        ///     not; an exception it throws is recorded as the problem.
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

        /// <summary>Fails the test, listing up to 60 failing cells, unless every cell held.</summary>
        /// <param name="what">What a cell is, for the message.</param>
        /// <exception cref="AssertFailedException">At least one cell failed, or no cell ran at all.</exception>
        internal void AssertNone(string what)
        {
            Assert.IsTrue(Cells > 0, $"no {what} cell ran");
            if (_problems.Count == 0)
            {
                Console.WriteLine($"{Cells} {what} cells checked, all held");
                return;
            }
            string shown = string.Join("\n  ", _problems.Take(60));
            string more = _problems.Count > 60 ? $"\n  … and {_problems.Count - 60} more" : "";
            Assert.Fail($"{_problems.Count} of {Cells} {what} cells failed:\n  {shown}{more}");
        }

        /// <summary>
        ///     The check for a call that must be refused: <c>null</c> when it throws <typeparamref name="TException"/> whose
        ///     message contains <paramref name="fragment"/>, else the problem. A call that unexpectedly succeeds has its result
        ///     disposed, so a wrong acceptance cannot also leak a view into the suite's leak gate.
        /// </summary>
        /// <typeparam name="TException">The exception the refusal must throw.</typeparam>
        /// <param name="call">The call.</param>
        /// <param name="fragment">Text the message must contain (null to skip the message check).</param>
        /// <returns><c>null</c> when refused as expected, else the problem.</returns>
        internal static string Refused<TException>(Func<object> call, string fragment) where TException : Exception
        {
            try
            {
                (call() as IDisposable)?.Dispose();
                return $"accepted; {typeof(TException).Name} expected";
            }
            catch (TException e)
            {
                return fragment == null || e.Message.Contains(fragment) ? null : $"{typeof(TException).Name} without \"{fragment}\": {e.Message}";
            }
        }
    }

    /// <summary>The NumSharp array layouts the matrix and property tests run the bridge over.</summary>
    internal static class Layouts
    {
        /// <summary>A C-contiguous array of the given shape holding <paramref name="value"/>(i) at flat index i.</summary>
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

        /// <summary>
        ///     Every named layout, each built fresh by its factory. The data views take rank 1 and 2 only, so the catalogue
        ///     covers those in depth (every slicing, stepping, reversing, transposing and broadcasting pattern) and keeps a
        ///     few other ranks for the refusals and for <c>ToVBuffer</c>, which flattens any rank.
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
                ("C 2-D (5,4)", () => B(5, 4)),
                ("C 3-D (2,3,4) (refused by the views)", () => B(2, 3, 4)),
                ("0-d scalar (refused by the views)", () => NDArray.Scalar(value(4))),
                ("empty (0,)", () => B(0)),
                ("empty (3,0)", () => B(3, 0)),
                ("empty (0,3)", () => B(0, 3)),
                ("empty slice at an offset a[5:5]", () => B(8)["5:5"]),
                ("empty column slice m[:, 3:3]", () => B(4, 5)[":, 3:3"]),
                ("empty row slice m[2:2]", () => B(4, 5)["2:2"]),
                ("last element a[-1:]", () => B(8)["-1:"]),
                ("offset slice a[2:5]", () => B(8)["2:5"]),
                ("row slice m[1:3]", () => B(4, 5)["1:3"]),
                ("column slice m[:, 1:3]", () => B(4, 5)[":, 1:3"]),
                ("single row m[2]", () => B(4, 5)["2"]),
                ("single column m[:, 2]", () => B(4, 5)[":, 2"]),
                ("single-row matrix m[2:3]", () => B(4, 5)["2:3"]),
                ("single-column matrix m[:, 2:3]", () => B(4, 5)[":, 2:3"]),
                ("stepped a[::2]", () => B(10)["::2"]),
                ("wide step a[::7]", () => B(50)["::7"]),
                ("stepped 2-D m[::2, 1::3]", () => B(6, 7)["::2, 1::3"]),
                ("transposed m.T", () => B(3, 4).T),
                ("F-contiguous", () => np.asfortranarray(B(3, 4))),
                ("F-contiguous column slice f[:, 1:3]", () => np.asfortranarray(B(3, 4))[":, 1:3"]),
                ("transposed slice m.T[1:3, ::2]", () => B(4, 5).T["1:3, ::2"]),
                ("expand_dims (1,6)", () => np.expand_dims(B(6), 0)),
                ("expand_dims (6,1)", () => np.expand_dims(B(6), 1)),
                ("broadcast row -> (3,4)", () => np.broadcast_to(B(4), new Shape(3, 4))),
                ("broadcast column -> (3,4)", () => np.broadcast_to(B(3, 1), new Shape(3, 4))),
                ("broadcast 0-d -> (2,3)", () => np.broadcast_to(NDArray.Scalar(value(5)), new Shape(2, 3))),
                ("broadcast 0-d -> (5,)", () => np.broadcast_to(NDArray.Scalar(value(5)), new Shape(5))),
                ("sliced broadcast", () => np.broadcast_to(B(4), new Shape(3, 4))["1:3, ::2"]),
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
                ("reversed both m[::-1, ::-1]", () => B(3, 4)["::-1, ::-1"]),
                ("reversed stepped a[::-2]", () => B(9)["::-2"]),
                ("reversed transposed m.T[::-1]", () => B(3, 4).T["::-1"]),
                ("view of a view m[1:][:, ::2]", () => B(5, 6)["1:"][":, ::2"]),
            };
        }
    }

    /// <summary>
    ///     The buffer pool's take/return balance: how many buffers an action took from NumSharp's pool and did not give back,
    ///     measured in a window in which no garbage collection ran (see the ONNX suite's twin for the protocol's reasons).
    /// </summary>
    internal static class Pool
    {
        /// <summary>Buffers taken so far (served from the pool, freshly allocated, or zero-allocated).</summary>
        private static long Taken => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;

        /// <summary>Buffers given back so far (kept in the pool or freed).</summary>
        private static long Given => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        /// <summary>
        ///     Runs <paramref name="action"/> once unmeasured (first-use allocations a static cache keeps are not its leak),
        ///     then in a GC-quiet window, and returns the buffers it took minus those it returned; a disturbed window is
        ///     retried, up to five times.
        /// </summary>
        /// <param name="action">The action to measure; it must dispose whatever it does not mean to keep.</param>
        /// <returns>The balance of the last undisturbed attempt (or of the fifth if every one was disturbed).</returns>
        internal static long Balance(Action action)
        {
            action();
            for (int attempt = 0; ; attempt++)
            {
                GcQuiescence.CollectFull();
                GcQuiescence.Epoch window = GcQuiescence.OpenWindow();
                long before = Taken - Given;
                action();
                long after = Taken - Given;
                if (GcQuiescence.Undisturbed(window) || attempt == 4)
                    return after - before;
            }
        }
    }

    /// <summary>
    ///     The bridge's export contract for ONE NumSharp view: what each verb must accept or refuse, and what an accepted
    ///     view must read as — through ML.NET's own cursors, through <c>ToVBuffer</c>, and back through the import verbs.
    /// </summary>
    internal static class ExportContract
    {
        /// <summary>
        ///     Runs every export check over <paramref name="view"/>, each verb its own cell. Complex has no ML.NET type; every
        ///     other dtype crosses as its ML.NET type (Char as <see cref="ushort"/>, Half as <see cref="float"/>, Decimal as
        ///     <see cref="double"/>).
        /// </summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name.</param>
        /// <param name="view">The view to export.</param>
        /// <param name="value">The type's value generator (for the snapshot-independence sentinel).</param>
        internal static void CheckAll<T>(Failures failures, string cell, NDArray view, Func<long, T> value) where T : unmanaged
        {
            switch (view.typecode)
            {
                case NPTypeCode.Char:
                    Check<T, ushort>(failures, cell, view, value);
                    return;
                case NPTypeCode.Half:
                    Check<T, float>(failures, cell, view, value);
                    return;
                case NPTypeCode.Decimal:
                    Check<T, double>(failures, cell, view, value);
                    return;
                case NPTypeCode.Complex:
                    CheckComplex(failures, cell, view);
                    return;
                default:
                    Check<T, T>(failures, cell, view, value);
                    return;
            }
        }

        /// <summary>Complex has no ML.NET type: every verb refuses it (a rank the views cannot take is refused for the rank first).</summary>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name.</param>
        /// <param name="view">The Complex view.</param>
        private static void CheckComplex(Failures failures, string cell, NDArray view)
        {
            bool rankOk = view.ndim == 1 || view.ndim == 2;
            string[] names = Names(view);
            failures.Check($"{cell} / AsDataView(column)", () => rankOk
                ? Failures.Refused<NotSupportedException>(() => view.AsDataView("F"), "omplex")
                : Failures.Refused<ArgumentException>(() => view.AsDataView("F"), "1-D or 2-D"));
            failures.Check($"{cell} / ToDataView(columns)", () => rankOk
                ? Failures.Refused<NotSupportedException>(() => view.ToDataView(names), "omplex")
                : Failures.Refused<ArgumentException>(() => view.ToDataView(names), "1-D or 2-D"));
            failures.Check($"{cell} / ToVBuffer", () => Failures.Refused<NotSupportedException>(() => view.ToVBuffer<double>(), "omplex"));
        }

        /// <summary>The export checks for a dtype that crosses as <typeparamref name="X"/>.</summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <typeparam name="X">The element type ML.NET sees.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name.</param>
        /// <param name="view">The view.</param>
        /// <param name="value">The type's value generator.</param>
        private static void Check<T, X>(Failures failures, string cell, NDArray view, Func<long, T> value) where T : unmanaged where X : unmanaged
        {
            T[] want = Oracle.Elements<T>(view);
            X[] crossed = Oracle.Cross<T, X>(want);
            byte[] crossedBytes = Oracle.Bytes(crossed);
            long[] dims = view.shape;
            NPTypeCode backCode = Oracle.CrossedCode(view.typecode);
            bool rankOk = dims.Length == 1 || dims.Length == 2;
            long rows = dims.Length == 2 ? dims[0] : 1;
            long width = dims.Length == 2 ? dims[1] : dims.Length == 1 ? dims[0] : 0;
            string[] names = Names(view);
            long scalarRows = dims.Length >= 1 ? dims[0] : 0;
            long scalarCols = dims.Length == 2 ? dims[1] : 1;

            foreach (bool snapshot in new[] { false, true })
            {
                string verb = snapshot ? "ToDataView" : "AsDataView";
                failures.Check($"{cell} / {verb}(column)", () =>
                {
                    if (!rankOk)
                        return Failures.Refused<ArgumentException>(() => snapshot ? view.ToDataView("F") : view.AsDataView("F"), "1-D or 2-D");
                    using NDArrayDataView dv = snapshot ? view.ToDataView("F") : view.AsDataView("F");
                    if (dv.GetRowCount() != rows)
                        return $"row count {dv.GetRowCount()}, expected {rows}";
                    if (dv.Schema["F"].Type is not VectorDataViewType vt || vt.Size != width || vt.ItemType.RawType != typeof(X))
                        return $"column type {dv.Schema["F"].Type}, expected a vector of {width} {typeof(X).Name}";
                    List<X[]> got = Oracle.VectorRows<X>(dv, "F");
                    if (got.Count != rows)
                        return $"the cursor produced {got.Count} rows, expected {rows}";
                    byte[] read = Oracle.Bytes(got.SelectMany(r => r).ToArray());
                    if (!Oracle.SameBytes(read, crossedBytes))
                        return "rows read through the cursor differ from NumSharp's elements";
                    if (snapshot && !SnapshotIsIndependent(view, value, () => Oracle.Bytes(Oracle.VectorRows<X>(dv, "F").SelectMany(r => r).ToArray()), crossedBytes))
                        return "the snapshot changed when the source did";
                    using NDArray back = dv.ToNDArray("F");
                    return Imported(back, backCode, new[] { rows, width }, crossedBytes);
                });

                failures.Check($"{cell} / {verb}(columns)", () =>
                {
                    if (!rankOk)
                        return Failures.Refused<ArgumentException>(() => snapshot ? view.ToDataView(names) : view.AsDataView(names), "1-D or 2-D");
                    if (scalarCols == 0)   // a (rows, 0) array has no column to name: the scalar shape cannot express it
                        return Failures.Refused<ArgumentException>(() => snapshot ? view.ToDataView(names) : view.AsDataView(names), "0 column(s)");
                    using NDArrayDataView dv = snapshot ? view.ToDataView(names) : view.AsDataView(names);
                    if (dv.GetRowCount() != scalarRows || dv.Schema.Count != scalarCols)
                        return $"{dv.GetRowCount()} rows x {dv.Schema.Count} columns, expected {scalarRows} x {scalarCols}";
                    List<X[]> got = Oracle.ScalarRows<X>(dv, names);
                    byte[] read = Oracle.Bytes(got.SelectMany(r => r).ToArray());
                    if (!Oracle.SameBytes(read, crossedBytes))
                        return "columns read through the cursor differ from NumSharp's elements";
                    if (snapshot && !SnapshotIsIndependent(view, value, () => Oracle.Bytes(Oracle.ScalarRows<X>(dv, names).SelectMany(r => r).ToArray()), crossedBytes))
                        return "the snapshot changed when the source did";
                    // The first column, back through the import verb. For a 1-D view the column IS the view, which must
                    // not be disposed here (it may own the buffer the next cells export).
                    using NDArray back = dv.ToNDArray(names[0]);
                    NDArray firstColumn = dims.Length == 2 ? view[":, 0"] : view;
                    try
                    {
                        byte[] columnBytes = Oracle.Bytes(Oracle.Cross<T, X>(Oracle.Elements<T>(firstColumn)));
                        return Imported(back, backCode, new[] { scalarRows }, columnBytes);
                    }
                    finally
                    {
                        if (!ReferenceEquals(firstColumn, view))
                            firstColumn.Dispose();
                    }
                });
            }

            failures.Check($"{cell} / ToVBuffer", () =>
            {
                VBuffer<X> buffer = view.ToVBuffer<X>();
                if (buffer.Length != want.Length || !buffer.IsDense)
                    return $"length {buffer.Length} dense {buffer.IsDense}, expected a dense buffer of {want.Length}";
                if (!Oracle.SameBytes(MemoryMarshal.AsBytes(buffer.GetValues()), crossedBytes))
                    return "buffer values differ from NumSharp's elements";
                using NDArray copy = buffer.ToNDArray();
                using NDArray view2 = buffer.AsNDArray();
                return Imported(copy, backCode, new long[] { want.Length }, crossedBytes)
                       ?? Imported(view2, backCode, new long[] { want.Length }, crossedBytes);
            });
        }

        /// <summary>
        ///     Writes a different value into the source's first element (when it can be written and has one) and checks the
        ///     snapshot still reads the original values, then restores the source.
        /// </summary>
        /// <typeparam name="T">The source's element type.</typeparam>
        /// <param name="source">The array the snapshot was taken from.</param>
        /// <param name="value">The type's value generator, for the sentinel.</param>
        /// <param name="readSnapshot">Reads the snapshot's values as bytes.</param>
        /// <param name="original">The snapshot's expected bytes.</param>
        /// <returns><c>true</c> when the snapshot is unaffected (or the source cannot be written).</returns>
        private static bool SnapshotIsIndependent<T>(NDArray source, Func<long, T> value, Func<byte[]> readSnapshot, byte[] original) where T : unmanaged
        {
            if (source.size == 0 || !source.Shape.IsWriteable)
                return true;
            T old = source.GetAtIndex<T>(0);
            source.SetAtIndex(Oracle.Different(old, value), 0);
            try
            {
                return Oracle.SameBytes(readSnapshot(), original);
            }
            finally
            {
                source.SetAtIndex(old, 0);
            }
        }

        /// <summary>One scalar-column name per column of a 1-D or 2-D array (<c>c0</c>, <c>c1</c>, …); a single name otherwise.</summary>
        /// <param name="nd">The array.</param>
        /// <returns>The names.</returns>
        internal static string[] Names(NDArray nd)
        {
            int count = nd.ndim == 2 ? (int)nd.shape[1] : 1;
            return Enumerable.Range(0, Math.Max(count, 1)).Select(i => "c" + i).ToArray();
        }

        /// <summary>Checks an imported array's dtype, shape and bytes (in logical C order).</summary>
        /// <param name="back">The imported array.</param>
        /// <param name="code">The expected dtype.</param>
        /// <param name="dims">The expected shape.</param>
        /// <param name="bytes">The expected bytes.</param>
        /// <returns><c>null</c> when all three match, else the problem.</returns>
        internal static string Imported(NDArray back, NPTypeCode code, long[] dims, byte[] bytes)
        {
            if (back.typecode != code)
                return $"dtype {back.typecode}, expected {code}";
            if (!back.shape.SequenceEqual(dims))
                return $"shape {Oracle.Describe(back.shape)}, expected {Oracle.Describe(dims)}";
            return Oracle.SameBytes(Oracle.LogicalBytes(back), bytes) ? null : "elements differ";
        }
    }
}
