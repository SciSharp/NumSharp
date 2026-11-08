using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
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
        ///     The value stored at a flat index. Different small indices give different values for every type with more
        ///     than two values, so an element read from the wrong place shows up; bool can only alternate, which is why
        ///     the write-through checks also write sentinels.
        /// </param>
        void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged;
    }

    /// <summary>The 15 NumSharp element types, each paired with a value generator, for <see cref="IDtypeCheck"/> sweeps.</summary>
    internal static class Dtypes
    {
        /// <summary>
        ///     Runs <paramref name="check"/> once per NumSharp element type. Every generator gives distinct values for at
        ///     least the first 250 indices (bool excepted), and the Decimal values are exact in float64, so the Decimal →
        ///     float64 conversion the bridge performs is checkable value for value.
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
    ///     Reference reads and index arithmetic that do not go through the bridge under test, so a test compares the bridge
    ///     against NumSharp's own view of the data (and against ONNX Runtime's own reads of a tensor).
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

        /// <summary>
        ///     The array's element bytes in logical C order, whatever its dtype: a C-order copy made by NumSharp's own
        ///     <c>copy()</c>, read raw. Used for arrays whose CLR element type the caller does not hold as a type
        ///     parameter (an import whose dtype is the crossed one, e.g. UInt16 for a Char source).
        /// </summary>
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

        /// <summary>A shape as text, e.g. <c>(3,0,4)</c>; ranks above 6 are summarized.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The description.</returns>
        internal static string Describe(long[] dims)
            => dims.Length > 6 ? $"rank-{dims.Length}" : "(" + string.Join(",", dims) + ")";

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

        /// <summary>
        ///     The dtype an array of <paramref name="code"/> reads back as after crossing to ONNX Runtime and back: Char
        ///     crosses as UInt16 and comes back as UInt16 (an ORT tensor carries no "these are characters" bit), Decimal is
        ///     converted to float64, every other supported dtype comes back as itself.
        /// </summary>
        /// <param name="code">The source dtype.</param>
        /// <returns>The imported dtype.</returns>
        internal static NPTypeCode CrossedCode(NPTypeCode code) => code switch
        {
            NPTypeCode.Char => NPTypeCode.UInt16,
            NPTypeCode.Decimal => NPTypeCode.Double,
            _ => code,
        };

        /// <summary>
        ///     The <c>casting</c> rule a sweep passes to <c>session.Run</c> so that an array of <paramref name="code"/> is
        ///     accepted by the identity model of its crossed dtype: the default <c>"safe"</c> for every dtype that crosses as
        ///     itself (and for Char, whose reinterpretation as UInt16 is a safe cast), <c>"same_kind"</c> for Decimal, whose
        ///     conversion to float64 loses digits and is therefore refused under <c>"safe"</c>.
        /// </summary>
        /// <param name="code">The source dtype.</param>
        /// <returns>The weakest rule that lets the array in.</returns>
        internal static string RunCasting(NPTypeCode code) => code == NPTypeCode.Decimal ? "same_kind" : "safe";

        /// <summary>The ONNX element type an array of <paramref name="code"/> crosses as (written out, not read from the bridge).</summary>
        /// <param name="code">The source dtype (not Complex).</param>
        /// <returns>The ONNX element type.</returns>
        internal static TensorElementType ElementType(NPTypeCode code) => code switch
        {
            NPTypeCode.Boolean => TensorElementType.Bool,
            NPTypeCode.Byte => TensorElementType.UInt8,
            NPTypeCode.SByte => TensorElementType.Int8,
            NPTypeCode.Int16 => TensorElementType.Int16,
            NPTypeCode.UInt16 => TensorElementType.UInt16,
            NPTypeCode.Char => TensorElementType.UInt16,
            NPTypeCode.Int32 => TensorElementType.Int32,
            NPTypeCode.UInt32 => TensorElementType.UInt32,
            NPTypeCode.Int64 => TensorElementType.Int64,
            NPTypeCode.UInt64 => TensorElementType.UInt64,
            NPTypeCode.Half => TensorElementType.Float16,
            NPTypeCode.Single => TensorElementType.Float,
            NPTypeCode.Double => TensorElementType.Double,
            NPTypeCode.Decimal => TensorElementType.Double,
            _ => throw new ArgumentOutOfRangeException(nameof(code), code, "no ONNX element type"),
        };

        /// <summary>
        ///     NumSharp's elements as the values that cross: the same bits for every dtype of the crossed type's size
        ///     (Char as UInt16, Half as ORT's Float16), and <see cref="decimal.ToDouble"/> of each value for Decimal — the
        ///     documented Decimal → float64 conversion, computed here without NumSharp.
        /// </summary>
        /// <typeparam name="T">The source element type.</typeparam>
        /// <typeparam name="X">The crossed element type.</typeparam>
        /// <param name="want">The source elements in logical C order.</param>
        /// <returns>The crossed values in the same order.</returns>
        internal static X[] Cross<T, X>(T[] want) where T : unmanaged where X : unmanaged
        {
            if (typeof(T) == typeof(decimal))
            {
                decimal[] source = (decimal[])(object)want;
                var converted = new double[source.Length];
                for (int i = 0; i < source.Length; i++)
                    converted[i] = decimal.ToDouble(source[i]);
                return (X[])(object)converted;
            }
            return MemoryMarshal.Cast<T, X>(want).ToArray();
        }

        /// <summary>A multi-index as the <c>int[]</c> a <see cref="DenseTensor{T}"/> indexer takes.</summary>
        /// <param name="index">The multi-index.</param>
        /// <returns>The same index as <c>int</c>s.</returns>
        internal static int[] Ints(long[] index) => Array.ConvertAll(index, i => (int)i);
    }

    /// <summary>
    ///     ONNX Runtime's own record of whether an <see cref="OrtValue"/> was disposed, read by reflection: the managed API
    ///     has no public "is disposed" member, and calling into a disposed value is not a safe way to ask. The bridge
    ///     promises to dispose a value in several places (an owning import of a tensor it copies, a copied run output), and
    ///     nothing else observes whether it did: a value that is merely dropped is freed later by ORT's finalizer, so the
    ///     suite's leak counters stay balanced either way.
    /// </summary>
    internal static class OrtProbe
    {
        /// <summary>The native handle field of <see cref="OrtValue"/>, which its <c>Dispose</c> sets to zero.</summary>
        private static readonly FieldInfo Handle = typeof(OrtValue).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic);

        /// <summary>
        ///     Has <paramref name="value"/> been disposed? The calling test is Inconclusive when the loaded ONNX Runtime no
        ///     longer has the field this reads (a rename in a future version), instead of failing or passing wrongly.
        /// </summary>
        /// <param name="value">The value to ask about.</param>
        /// <returns><c>true</c> when ORT released the native value.</returns>
        /// <exception cref="AssertInconclusiveException">ORT's <c>OrtValue</c> has no <c>IntPtr _handle</c> field.</exception>
        internal static bool IsDisposed(OrtValue value)
        {
            if (Handle is null || Handle.FieldType != typeof(IntPtr))
                Assert.Inconclusive("this ONNX Runtime's OrtValue has no private IntPtr _handle field; the disposal checks cannot read it.");
            return (IntPtr)Handle.GetValue(value) == IntPtr.Zero;
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

        /// <summary>
        ///     The check for a call that must be refused: <c>null</c> when it throws <typeparamref name="TException"/> whose
        ///     message contains <paramref name="fragment"/>, else the problem. A call that unexpectedly succeeds has its
        ///     result disposed, so a wrong acceptance cannot also leak a handle into the suite's leak gate.
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
        /// <summary>
        ///     A C-contiguous array of the given shape holding <paramref name="value"/>(i) at flat index i. A 1-D shape is the
        ///     array itself; any other shape is a reshaped view of it (the 1-D owner stays alive through the view).
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
        ///     Every named layout, each built fresh by its factory: contiguous at several ranks, 0-d, empty (also at an offset
        ///     and reversed), offset / row / column / single-row / single-column / stepped / wide-step slices, transposes and
        ///     permutations, Fortran order (also sliced), unit axes, broadcasts (row, column, 0-d, sliced, made writeable), a
        ///     diagonal, a read-only flag, negative strides, and ranks 40 and 70.
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
                ("F-contiguous 3-D", () => np.asfortranarray(B(2, 3, 4))),
                ("F-contiguous column slice f[:, 1:3]", () => np.asfortranarray(B(3, 4))[":, 1:3"]),
                ("transposed slice m.T[1:3, ::2]", () => B(4, 5).T["1:3, ::2"]),
                ("expand_dims (1,6)", () => np.expand_dims(B(6), 0)),
                ("expand_dims (2,1,3)", () => np.expand_dims(B(2, 3), 1)),
                ("unit axes (3,1)", () => B(3, 1)),
                ("unit axes (1,4,1)", () => B(1, 4, 1)),
                ("broadcast row -> (3,4)", () => np.broadcast_to(B(4), new Shape(3, 4))),
                ("broadcast column -> (3,4)", () => np.broadcast_to(B(3, 1), new Shape(3, 4))),
                ("broadcast 0-d -> (2,3)", () => np.broadcast_to(NDArray.Scalar(value(5)), new Shape(2, 3))),
                ("broadcast to one element (1,)", () => np.broadcast_to(NDArray.Scalar(value(5)), new Shape(1))),
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
                ("rank 40", () => B(HighRank(40))),
                ("rank 70", () => B(HighRank(70))),
            };
        }
    }

    /// <summary>
    ///     The buffer pool's take/return balance: how many buffers an action took from NumSharp's pool and did not give
    ///     back. Inside a window in which no garbage collection ran (so no finalizer returned an unrelated buffer), a leak
    ///     shows as a positive balance; an action whose every intermediate is disposed balances to exactly zero once its
    ///     own results are disposed too.
    /// </summary>
    internal static class Pool
    {
        /// <summary>Buffers taken so far (served from the pool, freshly allocated, or zero-allocated).</summary>
        private static long Taken => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;

        /// <summary>Buffers given back so far (kept in the pool or freed).</summary>
        private static long Given => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        /// <summary>
        ///     Runs <paramref name="action"/> in a GC-quiet window and returns the buffers it took minus those it returned.
        ///     The action runs once first, unmeasured: a first call can allocate a buffer some static cache keeps for the
        ///     life of the process (a type initializer, a lazily built table), which is not the action's leak. A window a
        ///     collection disturbed is retried (the action must be repeatable), up to five times.
        /// </summary>
        /// <param name="action">The action to measure; it must dispose whatever it does not mean to keep.</param>
        /// <returns>The balance of the last undisturbed attempt (or of the fifth attempt if every one was disturbed).</returns>
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

        /// <summary>
        ///     The buffers <paramref name="action"/> takes from the pool (served, freshly allocated or zero-allocated —
        ///     multi-gigabyte requests included, which bypass the pool and count as fresh allocations), whatever it gives
        ///     back. A refusal decided from the dtype and the shape alone must take NONE: a call that converts or copies
        ///     before refusing shows up here even when it disposes what it made. No warm-up and no GC window are needed:
        ///     finalizers only give buffers back, and the suite runs one test at a time.
        /// </summary>
        /// <param name="action">The action to measure (typically an asserted refusal).</param>
        /// <returns>The number of buffers taken while it ran.</returns>
        internal static long Takes(Action action)
        {
            long before = Taken;
            action();
            return Taken - before;
        }
    }

    /// <summary>
    ///     The bridge's export contract for ONE NumSharp view, checked against NumSharp's own reads and ONNX Runtime's own
    ///     reads: the matrix test runs it over every catalogued layout of every dtype, the property tests over random views.
    ///     Each verb is a cell of its own, and each mode-taking verb has a cell per <see cref="OnnxCodecMode"/>; a call the
    ///     contract says must refuse the view is checked for that refusal.
    /// </summary>
    internal static class ExportContract
    {
        /// <summary>The most elements whose writes are checked one by one per view.</summary>
        private const int MaxIndividualWrites = 16;

        /// <summary>The three modes, for the cells that must hold in every one of them.</summary>
        private static readonly OnnxCodecMode[] Modes = { OnnxCodecMode.Auto, OnnxCodecMode.View, OnnxCodecMode.Copy };

        /// <summary>
        ///     Runs every export check over <paramref name="view"/>. Complex has no ONNX type, so every verb must refuse it
        ///     in every mode (a copy cannot give it one); every other dtype is checked through its crossed element type
        ///     (Char as <see cref="ushort"/>, Half as ORT's <see cref="Float16"/>, Decimal converted to <see cref="double"/>).
        /// </summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name (dtype, layout or random-case description).</param>
        /// <param name="view">The view to export.</param>
        /// <param name="value">The type's value generator (for write-through sentinels).</param>
        internal static void CheckAll<T>(Failures failures, string cell, NDArray view, Func<long, T> value) where T : unmanaged
        {
            switch (view.typecode)
            {
                case NPTypeCode.Complex:
                    foreach (OnnxCodecMode mode in Modes)
                    {
                        failures.Check($"{cell} / AsOrtValue({mode})", () => Failures.Refused<NotSupportedException>(() => view.AsOrtValue(mode), "complex"));
                        failures.Check($"{cell} / AsDenseTensor({mode})", () => Failures.Refused<NotSupportedException>(() => view.AsDenseTensor<double>(mode), "complex"));
                    }
                    failures.Check($"{cell} / ToOrtValue", () => Failures.Refused<NotSupportedException>(() => view.ToOrtValue(), "complex"));
                    failures.Check($"{cell} / ToDenseTensor", () => Failures.Refused<NotSupportedException>(() => view.ToDenseTensor<double>(), "complex"));
                    return;
                case NPTypeCode.Char:
                    Check<T, ushort>(failures, cell, view, value);
                    return;
                case NPTypeCode.Half:
                    Check<T, Float16>(failures, cell, view, value);
                    return;
                case NPTypeCode.Decimal:
                    Check<T, double>(failures, cell, view, value);
                    return;
                default:
                    Check<T, T>(failures, cell, view, value);
                    return;
            }
        }

        /// <summary>The export checks for a dtype that crosses as <typeparamref name="X"/>.</summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <typeparam name="X">The element type ONNX Runtime sees.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name.</param>
        /// <param name="view">The view.</param>
        /// <param name="value">The value generator.</param>
        private static void Check<T, X>(Failures failures, string cell, NDArray view, Func<long, T> value) where T : unmanaged where X : unmanaged
        {
            NPTypeCode code = view.typecode;
            T[] want = Oracle.Elements<T>(view);
            X[] crossed = Oracle.Cross<T, X>(want);
            byte[] crossedBytes = Oracle.Bytes(crossed);
            long[] dims = view.shape;
            bool converted = code == NPTypeCode.Decimal;
            TensorElementType elementType = Oracle.ElementType(code);
            NPTypeCode backCode = Oracle.CrossedCode(code);

            // When ONNX Runtime can read the view's own buffer: the dtype crosses bit for bit (every dtype here except
            // Decimal, which is converted) and the view is C-contiguous. Then Auto and View share it; every other view
            // crosses as a copy under Auto and Copy and is refused under View. (This is the test's own statement of the
            // rule, computed from NumSharp's flags, not read from the bridge.)
            bool viewPossible = !converted && view.Shape.IsContiguous;
            bool writeThrough = viewPossible && view.Shape.IsWriteable && want.Length > 0 && !Oracle.HasZeroStride(dims, view.Shape.strides);
            string refusal = converted ? "Decimal" : "not C-contiguous";

            failures.Check($"{cell} / AsOrtValue", () =>
            {
                using OrtTensor h = view.AsOrtValue();
                return HandleIs(h.IsView, h.Source, view, viewPossible)
                       ?? OrtTensorHolds(h.Value, elementType, dims, crossedBytes)
                       ?? (viewPossible
                           ? (writeThrough ? WritesThroughOrt(h.Value, view, want, value) : null)
                           : IndependentOfSource(h.Value, view, want));
            });

            failures.Check($"{cell} / AsOrtValue(View)", () =>
            {
                if (!viewPossible)
                    return Failures.Refused<InvalidOperationException>(() => view.AsOrtValue(OnnxCodecMode.View), refusal);
                using OrtTensor h = view.AsOrtValue(OnnxCodecMode.View);
                return HandleIs(h.IsView, h.Source, view, expectedView: true)
                       ?? OrtTensorHolds(h.Value, elementType, dims, crossedBytes)
                       ?? (writeThrough ? WritesThroughOrt(h.Value, view, want, value) : null);
            });

            failures.Check($"{cell} / AsOrtValue(Copy)", () =>
            {
                using OrtTensor h = view.AsOrtValue(OnnxCodecMode.Copy);
                return HandleIs(h.IsView, h.Source, view, expectedView: false)
                       ?? OrtTensorHolds(h.Value, elementType, dims, crossedBytes)
                       ?? IndependentOfSource(h.Value, view, want);
            });

            failures.Check($"{cell} / ToOrtValue", () =>
            {
                using OrtValue v = view.ToOrtValue();
                return OrtTensorHolds(v, elementType, dims, crossedBytes) ?? IndependentOfSource(v, view, want);
            });

            failures.Check($"{cell} / AsDenseTensor", () =>
            {
                using OrtTensor<X> h = view.AsDenseTensor<X>();
                return HandleIs(h.IsView, h.Source, view, viewPossible)
                       ?? DenseHolds(h.Tensor, dims, crossed, reversed: false)
                       ?? (viewPossible
                           ? (writeThrough ? WritesThroughDense(h.Tensor, view, want, value, dims) : null)
                           : IndependentOfSource(h.Tensor, view, want));
            });

            failures.Check($"{cell} / AsDenseTensor(View)", () =>
            {
                if (!viewPossible)
                    return Failures.Refused<InvalidOperationException>(() => view.AsDenseTensor<X>(OnnxCodecMode.View), refusal);
                using OrtTensor<X> h = view.AsDenseTensor<X>(OnnxCodecMode.View);
                return HandleIs(h.IsView, h.Source, view, expectedView: true)
                       ?? DenseHolds(h.Tensor, dims, crossed, reversed: false)
                       ?? (writeThrough ? WritesThroughDense(h.Tensor, view, want, value, dims) : null);
            });

            failures.Check($"{cell} / AsDenseTensor(Copy)", () =>
            {
                using OrtTensor<X> h = view.AsDenseTensor<X>(OnnxCodecMode.Copy);
                return HandleIs(h.IsView, h.Source, view, expectedView: false)
                       ?? DenseHolds(h.Tensor, dims, crossed, reversed: false)
                       ?? IndependentOfSource(h.Tensor, view, want);
            });

            failures.Check($"{cell} / ToDenseTensor", () => DenseHolds(view.ToDenseTensor<X>(), dims, crossed, reversed: false));

            failures.Check($"{cell} / ToOrtValue->ToNDArray", () =>
            {
                using OrtValue v = view.ToOrtValue();
                using NDArray back = v.ToNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            failures.Check($"{cell} / ToOrtValue->AsNDArray", () =>
            {
                using OrtValue v = view.ToOrtValue();
                using NDArray back = v.AsNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            // Every layout round-trips through the handle verbs now: a view that cannot be shared crosses as a copy.
            failures.Check($"{cell} / AsOrtValue->AsNDArray", () =>
            {
                using OrtTensor h = view.AsOrtValue();
                using NDArray back = h.Value.AsNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            failures.Check($"{cell} / ToDenseTensor->ToNDArray", () =>
            {
                using NDArray back = view.ToDenseTensor<X>().ToNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            failures.Check($"{cell} / ToDenseTensor->AsNDArray", () =>
            {
                using NDArray back = view.ToDenseTensor<X>().AsNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            failures.Check($"{cell} / AsDenseTensor->ToNDArray", () =>
            {
                using OrtTensor<X> h = view.AsDenseTensor<X>();
                using NDArray back = h.Tensor.ToNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });

            failures.Check($"{cell} / AsDenseTensor->AsNDArray", () =>
            {
                using OrtTensor<X> h = view.AsDenseTensor<X>();
                using NDArray back = h.Tensor.AsNDArray();
                return Imported(back, backCode, dims, crossedBytes);
            });
        }

        /// <summary>
        ///     Checks what an export handle says about itself: <c>IsView</c> must be what the layout rule predicts, and
        ///     <c>Source</c> must be the caller's array exactly when the handle is a view (a copy's <c>Source</c> is the
        ///     C-contiguous temporary ORT reads).
        /// </summary>
        /// <param name="isView">The handle's <c>IsView</c>.</param>
        /// <param name="source">The handle's <c>Source</c>.</param>
        /// <param name="view">The array that was exported.</param>
        /// <param name="expectedView">Whether the handle must be a view.</param>
        /// <returns><c>null</c> when all hold, else the problem.</returns>
        private static string HandleIs(bool isView, NDArray source, NDArray view, bool expectedView)
        {
            if (isView != expectedView)
                return $"IsView {isView}, expected {expectedView}";
            if (ReferenceEquals(source, view) != expectedView)
                return expectedView ? "a view's Source is not the exported array" : "a copy's Source is the exported array itself";
            if (!source.Shape.IsContiguous)
                return "the array ORT reads (Source) is not C-contiguous";
            return null;
        }

        /// <summary>
        ///     Checks that an ORT tensor is an independent copy: every byte of it is overwritten, and NumSharp's view must
        ///     still hold its own elements.
        /// </summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <param name="v">The tensor that must not share memory with <paramref name="view"/>.</param>
        /// <param name="view">The NumSharp view it was made from.</param>
        /// <param name="want">The view's elements before the overwrite.</param>
        /// <returns><c>null</c> when the view is unchanged (or empty), else the problem.</returns>
        private static string IndependentOfSource<T>(OrtValue v, NDArray view, T[] want) where T : unmanaged
        {
            if (want.Length == 0)
                return null;
            v.GetTensorMutableRawData().Fill(0xA5);
            return Oracle.Bytes(Oracle.Elements<T>(view)).AsSpan().SequenceEqual(Oracle.Bytes(want)) ? null : "the tensor shares memory with the source, an independent copy was expected";
        }

        /// <summary>
        ///     Checks that a <see cref="DenseTensor{T}"/> is an independent copy: every byte of its buffer is overwritten,
        ///     and NumSharp's view must still hold its own elements.
        /// </summary>
        /// <typeparam name="T">The view's element type.</typeparam>
        /// <typeparam name="X">The tensor's element type.</typeparam>
        /// <param name="t">The tensor that must not share memory with <paramref name="view"/>.</param>
        /// <param name="view">The NumSharp view it was made from.</param>
        /// <param name="want">The view's elements before the overwrite.</param>
        /// <returns><c>null</c> when the view is unchanged (or empty), else the problem.</returns>
        private static string IndependentOfSource<T, X>(DenseTensor<X> t, NDArray view, T[] want) where T : unmanaged where X : unmanaged
        {
            if (want.Length == 0)
                return null;
            MemoryMarshal.AsBytes(t.Buffer.Span).Fill(0xA5);
            return Oracle.Bytes(Oracle.Elements<T>(view)).AsSpan().SequenceEqual(Oracle.Bytes(want)) ? null : "the tensor shares memory with the source, an independent copy was expected";
        }

        /// <summary>Checks an ORT tensor's element type, shape, element count and bytes (read through ORT's own API).</summary>
        /// <param name="v">The tensor.</param>
        /// <param name="type">The expected element type.</param>
        /// <param name="dims">The expected shape.</param>
        /// <param name="bytes">The expected bytes in row-major order.</param>
        /// <returns><c>null</c> when all hold, else the problem.</returns>
        internal static string OrtTensorHolds(OrtValue v, TensorElementType type, long[] dims, byte[] bytes)
        {
            OrtTensorTypeAndShapeInfo info = v.GetTensorTypeAndShape();
            if (info.ElementDataType != type)
                return $"element type {info.ElementDataType}, expected {type}";
            if (!info.Shape.SequenceEqual(dims))
                return $"tensor shape {Oracle.Describe(info.Shape)}, expected {Oracle.Describe(dims)}";
            long size = dims.Aggregate(1L, (a, d) => a * d);
            if (info.ElementCount != size)
                return $"element count {info.ElementCount}, expected {size}";
            if (size == 0)
                return null;
            return Oracle.SameBytes(v.GetTensorMutableRawData(), bytes) ? null : "tensor bytes differ from NumSharp's elements";
        }

        /// <summary>
        ///     Checks a <see cref="DenseTensor{T}"/>'s dimensions, stride order and EVERY element, read through the tensor's
        ///     own strided indexer (which applies <c>reverseStride</c> itself).
        /// </summary>
        /// <typeparam name="X">The tensor's element type.</typeparam>
        /// <param name="t">The tensor.</param>
        /// <param name="dims">The expected shape.</param>
        /// <param name="crossed">The expected values in logical C order.</param>
        /// <param name="reversed">Whether the tensor must be column-major.</param>
        /// <returns><c>null</c> when all hold, else the problem.</returns>
        internal static string DenseHolds<X>(DenseTensor<X> t, long[] dims, X[] crossed, bool reversed) where X : unmanaged
        {
            long[] got = t.Dimensions.ToArray().Select(d => (long)d).ToArray();
            if (!got.SequenceEqual(dims))
                return $"tensor dimensions {Oracle.Describe(got)}, expected {Oracle.Describe(dims)}";
            if (t.IsReversedStride != reversed)
                return $"IsReversedStride {t.IsReversedStride}, expected {reversed}";
            foreach (long[] index in Oracle.Indices(dims))
                if (!Oracle.SameBytes(t[Oracle.Ints(index)], crossed[Oracle.Flat(index, dims)]))
                    return $"tensor[{string.Join(",", index)}] differs from NumSharp's element";
            return null;
        }

        /// <summary>
        ///     Writes through the ORT tensor's own mutable span and checks NumSharp sees exactly that write: each index (up to
        ///     a cap) is written in turn with a sentinel, and ONLY the NumSharp element at that logical index may change.
        /// </summary>
        /// <typeparam name="T">The element type (the crossed type has the same size: no conversion happened).</typeparam>
        /// <param name="v">The shared tensor.</param>
        /// <param name="view">The NumSharp view it shares.</param>
        /// <param name="want">NumSharp's elements before any write.</param>
        /// <param name="value">The value generator, for sentinels.</param>
        /// <returns><c>null</c> when every write landed where it should, else the problem.</returns>
        private static string WritesThroughOrt<T>(OrtValue v, NDArray view, T[] want, Func<long, T> value) where T : unmanaged
        {
            Span<T> span = MemoryMarshal.Cast<byte, T>(v.GetTensorMutableRawData());
            for (int k = 0; k < Math.Min(want.Length, MaxIndividualWrites); k++)
            {
                T old = span[k];
                T sentinel = Oracle.Different(old, value);
                span[k] = sentinel;
                T[] now = Oracle.Elements<T>(view);
                span[k] = old;
                if (!Oracle.SameBytes(now[k], sentinel))
                    return $"a write through the ORT tensor at {k} is not visible at that NumSharp element";
                for (int i = 0; i < now.Length; i++)
                    if (i != k && !Oracle.SameBytes(now[i], want[i]))
                        return $"a write through the ORT tensor at {k} also changed NumSharp element {i}";
            }
            return null;
        }

        /// <summary>
        ///     Writes through a shared <see cref="DenseTensor{T}"/>'s indexer and checks NumSharp sees exactly that write (see
        ///     <see cref="WritesThroughOrt{T}"/>); the indexer applies the tensor's stride order, so a Fortran-ordered share is
        ///     checked in logical order too.
        /// </summary>
        /// <typeparam name="T">The NumSharp element type.</typeparam>
        /// <typeparam name="X">The tensor's element type (same size as <typeparamref name="T"/>).</typeparam>
        /// <param name="t">The shared tensor.</param>
        /// <param name="view">The NumSharp view it shares.</param>
        /// <param name="want">NumSharp's elements before any write.</param>
        /// <param name="value">The value generator, for sentinels.</param>
        /// <param name="dims">The shape.</param>
        /// <returns><c>null</c> when every write landed where it should, else the problem.</returns>
        private static string WritesThroughDense<T, X>(DenseTensor<X> t, NDArray view, T[] want, Func<long, T> value, long[] dims) where T : unmanaged where X : unmanaged
        {
            foreach (long[] index in Oracle.Indices(dims).Take(MaxIndividualWrites))
            {
                int[] at = Oracle.Ints(index);
                long flat = Oracle.Flat(index, dims);
                X old = t[at];
                T sentinelT = Oracle.Different(MemoryMarshal.Cast<X, T>(new[] { old })[0], value);
                X sentinel = MemoryMarshal.Cast<T, X>(new[] { sentinelT })[0];
                t[at] = sentinel;
                T[] now = Oracle.Elements<T>(view);
                t[at] = old;
                if (!Oracle.SameBytes(now[flat], sentinelT))
                    return $"a write through tensor[{string.Join(",", index)}] is not visible at that NumSharp element";
                for (long i = 0; i < now.LongLength; i++)
                    if (i != flat && !Oracle.SameBytes(now[i], want[i]))
                        return $"a write through tensor[{string.Join(",", index)}] also changed NumSharp element {i}";
            }
            return null;
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
