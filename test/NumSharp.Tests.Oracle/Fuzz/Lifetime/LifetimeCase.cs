using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     One case of the lifetime oracle (<c>test/oracle/gen_lifetime_oracle.py</c>): a call, its operands as
    ///     RECIPES, which operands the stressed repetitions build afresh, and NumPy 2.4.2's answer.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The operands are recipes rather than bytes because they must be large (2-8 MB each): a kernel has to run long
    ///     enough for a forced collection to land inside it, and storing every operand's bytes would cost about 1 GB of
    ///     corpus. Both sides build the same values from a recipe (<see cref="LifetimeOperand.BuildBytes"/> mirrors the
    ///     generator's <c>build</c> statement for statement).
    ///     </para>
    ///     <para>
    ///     A case is deserialized once per child run; its operand bytes are built lazily, once, and reused by every
    ///     repetition (<see cref="LifetimeOperand.Make"/> copies them into a fresh array).
    ///     </para>
    /// </remarks>
    internal sealed class LifetimeCase
    {
        /// <summary>The case id, <c>lifetime/&lt;op&gt;/&lt;tag&gt;/&lt;dtype&gt;/&lt;n&gt;</c>.</summary>
        public string Id { get; set; }

        /// <summary>The op key <see cref="LifetimeOps.Invoke"/> dispatches on.</summary>
        public string Op { get; set; }

        /// <summary>The operands in call order.</summary>
        public LifetimeOperand[] Args { get; set; }

        /// <summary>The call's parameters (axis, mode, dtype, decimals); never null after <see cref="Parse"/>.</summary>
        public Dictionary<string, JsonElement> Params { get; set; }

        /// <summary>
        ///     True when NumSharp reproduces NumPy's VALUE bit for bit here. False for results whose bits depend on an
        ///     algorithm NumSharp does not share (managed LU and Gram matrices, an introselect arrangement): the replay
        ///     then compares only dtype and shape with NumPy, and still holds every stressed repetition to the reference.
        /// </summary>
        public bool Parity { get; set; }

        /// <summary>
        ///     True when the value depends on the platform libm or the FFT twiddles of win-amd64 NumPy: the replay
        ///     compares it with NumPy only on a Windows x64 host.
        /// </summary>
        public bool Host { get; set; }

        /// <summary>NumPy's answer.</summary>
        public LifetimeExpected Expected { get; set; }

        /// <summary>The documented gap id of a carved case (<c>lifetime_gaps.jsonl</c>); null for a gate case.</summary>
        public string Gap { get; set; }

        /// <summary>The case's line index in its file (set by the reader, not serialized).</summary>
        public int Index { get; set; }

        /// <summary>The corpus is written by Python in lower case; the members are PascalCase.</summary>
        private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

        /// <summary>Parses one corpus line.</summary>
        /// <param name="line">The JSON text of the case.</param>
        /// <param name="index">The line index (zero-based) the case is reported under.</param>
        /// <returns>The case, with a non-null <see cref="Params"/>.</returns>
        /// <exception cref="JsonException">The line is not a well-formed case.</exception>
        public static LifetimeCase Parse(string line, int index)
        {
            var c = JsonSerializer.Deserialize<LifetimeCase>(line, Json)
                    ?? throw new JsonException("a corpus line deserialized to null");
            c.Params ??= new Dictionary<string, JsonElement>();
            c.Index = index;
            return c;
        }

        /// <summary>
        ///     Reads every case of a lifetime corpus file.
        /// </summary>
        /// <param name="path">The file's full path.</param>
        /// <returns>The cases in file order (blank lines skipped).</returns>
        /// <exception cref="FileNotFoundException">The file does not exist (a corpus copy regression).</exception>
        public static List<LifetimeCase> ReadAll(string path)
        {
            var cases = new List<LifetimeCase>();
            int index = 0;
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                cases.Add(Parse(line, index++));
            }
            return cases;
        }

        /// <summary>
        ///     The full path of a lifetime corpus file in the test output (<c>Fuzz/Lifetime/corpus/</c>, copied by the
        ///     csproj; NOT <c>Fuzz/corpus/</c>, whose generic sweeps expect the ordinary case schema).
        /// </summary>
        /// <param name="fileName">The file name, e.g. <c>lifetime.jsonl</c>.</param>
        /// <returns>The absolute path.</returns>
        public static string CorpusPath(string fileName)
            => Path.Combine(AppContext.BaseDirectory, "Fuzz", "Lifetime", "corpus", fileName);

        /// <summary>Reads an integer parameter, or the default when it is absent or JSON null.</summary>
        /// <param name="name">The parameter name.</param>
        /// <param name="fallback">The value when absent.</param>
        /// <returns>The parameter value.</returns>
        public int? IntParam(string name, int? fallback = null)
            => Params.TryGetValue(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

        /// <summary>Reads an integer-array parameter (e.g. the <c>dims</c> of <c>unravel_index</c>).</summary>
        /// <param name="name">The parameter name.</param>
        /// <returns>The array.</returns>
        /// <exception cref="KeyNotFoundException">The case has no such parameter.</exception>
        public int[] IntArrayParam(string name)
            => Params[name].EnumerateArray().Select(e => e.GetInt32()).ToArray();

        /// <summary>Reads a string parameter, or the default when it is absent or JSON null.</summary>
        /// <param name="name">The parameter name.</param>
        /// <param name="fallback">The value when absent.</param>
        /// <returns>The parameter value.</returns>
        public string StringParam(string name, string fallback = null)
            => Params.TryGetValue(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : fallback;
    }

    /// <summary>NumPy's recorded answer: one item, or a tuple of items.</summary>
    internal sealed class LifetimeExpected
    {
        /// <summary><c>array</c>, <c>bytes</c> or <c>tuple</c>.</summary>
        public string Kind { get; set; }

        /// <summary>NumPy's <c>dtype.str</c> of an array item (<c>&lt;f8</c>, <c>|b1</c>, <c>&lt;U6</c>).</summary>
        public string Dtype { get; set; }

        /// <summary>The shape of an array item (<c>[]</c> for a 0-d result or a scalar).</summary>
        public long[] Shape { get; set; }

        /// <summary>The length of a <c>bytes</c> item.</summary>
        public long Len { get; set; }

        /// <summary>The lower-case hex SHA-256 of the item's C-order bytes.</summary>
        public string Sha256 { get; set; }

        /// <summary>The items of a <c>tuple</c> result.</summary>
        public LifetimeExpected[] Items { get; set; }
    }

    /// <summary>
    ///     One operand: a recipe, the base shape, a view, and whether the stressed repetitions build it afresh.
    /// </summary>
    internal sealed class LifetimeOperand
    {
        /// <summary>The NumPy dtype name (<c>U6</c> for six-character strings).</summary>
        public string Dtype { get; set; }

        /// <summary>The BASE shape; <see cref="View"/> is taken from it.</summary>
        public long[] Shape { get; set; }

        /// <summary>The view applied to the base: empty, <c>step2</c>, <c>rev</c>, <c>T</c>, <c>colstep</c>, <c>off1</c>.</summary>
        public string View { get; set; }

        /// <summary>
        ///     True when every stressed repetition builds this operand afresh inside the call expression (the operand
        ///     under test); false when the repetitions pass one held instance (the case attributes a gap to another
        ///     operand).
        /// </summary>
        public bool Temp { get; set; }

        /// <summary>The value recipe (see <see cref="BuildBytes"/>).</summary>
        public Dictionary<string, JsonElement> Recipe { get; set; }

        /// <summary>The base buffer's bytes, built once per case by <see cref="BuildBytes"/> (see <see cref="ReleaseBytes"/>).</summary>
        private byte[] _bytes;

        /// <summary>The resolved <see cref="Dtype"/> (resolved once: <see cref="np.dtype(string)"/> parses the name).</summary>
        private DType _dtype;

        /// <summary>The resolved dtype descriptor.</summary>
        public DType Descr => _dtype ??= np.dtype(Dtype);

        /// <summary>The base buffer's byte size: the pool bucket a temporary of this operand is drawn from.</summary>
        public long ByteSize => ElementCount * Descr.itemsize;

        /// <summary>The number of elements of the base.</summary>
        public long ElementCount
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= d;
                return n;
            }
        }

        /// <summary>
        ///     Drops the cached recipe bytes. The child calls it after each case: a corpus of several hundred cases keeps
        ///     every case object alive, and their cached 2-8 MB byte arrays would grow the managed heap into gigabytes —
        ///     every forced full collection would then pay to mark and sweep them, and the stress thread would collect
        ///     less often.
        /// </summary>
        public void ReleaseBytes() => _bytes = null;

        /// <summary>
        ///     Builds the operand: a fresh array allocated from the pool (<see cref="np.empty(Shape, DType, string)"/>) holding
        ///     the recipe's bytes, with the view applied.
        /// </summary>
        /// <returns>
        ///     A new array the CALLER owns. Nothing else references it, so when the caller passes it straight into a
        ///     call, the call's parameter is its only reference — the situation the lifetime replay tests.
        /// </returns>
        /// <remarks>
        ///     <c>NoInlining</c> keeps the base local of this frame out of the caller's frame: after the view is taken
        ///     the base is reachable only through the view's storage, exactly as a caller's temporary view would be.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public unsafe NDArray Make()
        {
            var bytes = _bytes ??= BuildBytes();
            var a = np.empty(new Shape(Shape), Descr);
            fixed (byte* src = bytes)
                Buffer.MemoryCopy(src, (void*)a.Address, bytes.Length, bytes.Length);
            return ApplyView(a);
        }

        /// <summary>Applies <see cref="View"/> to a freshly built base.</summary>
        /// <param name="a">The base array.</param>
        /// <returns>The base itself, or a view of it.</returns>
        /// <exception cref="InvalidDataException">The view name is not one the generator writes.</exception>
        private NDArray ApplyView(NDArray a) => View switch
        {
            null or "" => a,
            "step2" => a["::2"],
            "rev" => a["::-1"],
            "T" => a.T,
            "colstep" => a[":, ::2"],
            "off1" => a["1:"],
            _ => throw new InvalidDataException($"unknown lifetime view '{View}'")
        };

        /// <summary>
        ///     Builds the base buffer's C-order bytes from the recipe, statement for statement as the generator's
        ///     <c>build</c> does.
        /// </summary>
        /// <returns>The bytes (<see cref="ByteSize"/> of them).</returns>
        /// <exception cref="InvalidDataException">The recipe kind or the dtype is one the generator never writes.</exception>
        /// <remarks>
        ///     <c>mod</c>: <c>q = (start + i*step) % mod</c> in 64-bit integers, <c>value = q*scale + off</c> in float64
        ///     (exact by construction: the generator picks power-of-two scales and small offsets), then bool
        ///     <c>q == 0</c>, integer dtypes <c>(long)value</c>, float dtypes the value, strings <c>str((long)value)</c>.
        ///     <c>eye</c> adds <c>diag</c> on the main diagonal; <c>cmod</c> builds the real and imaginary parts from two
        ///     progressions; <c>perm</c> is the index progression <c>(start + i*step) % n</c>.
        /// </remarks>
        public byte[] BuildBytes()
        {
            long n = ElementCount;
            string kind = Recipe["k"].GetString();
            var bytes = new byte[n * Descr.itemsize];
            long R(string k) => Recipe[k].GetInt64();
            double D(string k) => Recipe[k].GetDouble();
            switch (kind)
            {
                case "perm":
                {
                    long start = R("start"), step = R("step"), m = R("n");
                    for (long i = 0; i < n; i++)
                        WriteInteger(bytes, i, (start + i * step) % m);
                    break;
                }
                case "cmod":
                {
                    if (Dtype != "complex128")
                        throw new InvalidDataException($"cmod recipe for dtype {Dtype}");
                    long s1 = R("start"), t1 = R("step"), m1 = R("mod"), s2 = R("start2"), t2 = R("step2"), m2 = R("mod2");
                    double c1 = D("scale"), c2 = D("scale2");
                    for (long i = 0; i < n; i++)
                    {
                        double re = ((s1 + i * t1) % m1) * c1, im = ((s2 + i * t2) % m2) * c2;
                        BitConverter.TryWriteBytes(bytes.AsSpan((int)(i * 16)), re);
                        BitConverter.TryWriteBytes(bytes.AsSpan((int)(i * 16 + 8)), im);
                    }
                    break;
                }
                case "mod":
                case "eye":
                {
                    long start = R("start"), step = R("step"), m = R("mod");
                    double scale = D("scale"), off = D("off");
                    double diag = kind == "eye" ? D("diag") : 0.0;
                    long side = kind == "eye" ? Shape[0] : 0;
                    for (long i = 0; i < n; i++)
                    {
                        long q = (start + i * step) % m;
                        double v = q * scale + off;
                        if (kind == "eye" && i / side == i % side)
                            v += diag;
                        WriteValue(bytes, i, q, v);
                    }
                    break;
                }
                default:
                    throw new InvalidDataException($"unknown lifetime recipe kind '{kind}'");
            }
            return bytes;
        }

        /// <summary>Writes one integer element (the <c>perm</c> recipe) in the operand's integer dtype.</summary>
        private void WriteInteger(byte[] bytes, long i, long value)
        {
            switch (Dtype)
            {
                case "int64": BitConverter.TryWriteBytes(bytes.AsSpan((int)(i * 8)), value); break;
                case "int32": BitConverter.TryWriteBytes(bytes.AsSpan((int)(i * 4)), (int)value); break;
                default: throw new InvalidDataException($"perm recipe for dtype {Dtype}");
            }
        }

        /// <summary>
        ///     Writes one element of the <c>mod</c> / <c>eye</c> recipe: its exact float64 <paramref name="value"/>
        ///     converted to the operand's dtype (bool from <paramref name="q"/>, strings as the integer's decimal text).
        /// </summary>
        private void WriteValue(byte[] bytes, long i, long q, double value)
        {
            int size = Descr.itemsize;
            var span = bytes.AsSpan((int)(i * size), size);
            switch (Dtype)
            {
                case "bool": span[0] = (byte)(q == 0 ? 1 : 0); break;
                case "int8": span[0] = unchecked((byte)(sbyte)(long)value); break;
                case "uint8": span[0] = unchecked((byte)(long)value); break;
                case "int16": BitConverter.TryWriteBytes(span, unchecked((short)(long)value)); break;
                case "uint16": BitConverter.TryWriteBytes(span, unchecked((ushort)(long)value)); break;
                case "int32": BitConverter.TryWriteBytes(span, unchecked((int)(long)value)); break;
                case "uint32": BitConverter.TryWriteBytes(span, unchecked((uint)(long)value)); break;
                case "int64": BitConverter.TryWriteBytes(span, (long)value); break;
                case "uint64": BitConverter.TryWriteBytes(span, unchecked((ulong)(long)value)); break;
                case "float16": BitConverter.TryWriteBytes(span, (Half)value); break;
                case "float32": BitConverter.TryWriteBytes(span, (float)value); break;
                case "float64": BitConverter.TryWriteBytes(span, value); break;
                default:
                    if (Dtype.StartsWith("U", StringComparison.Ordinal))
                    {
                        // NumPy's '<U<n>': n UTF-32LE code units, zero padded (NumSharp stores the same layout).
                        string text = ((long)value).ToString(CultureInfo.InvariantCulture);
                        int width = size / 4;
                        if (text.Length > width)
                            throw new InvalidDataException($"recipe value {text} does not fit {Dtype}");
                        for (int c = 0; c < text.Length; c++)
                            BitConverter.TryWriteBytes(span.Slice(c * 4, 4), (int)text[c]);
                        break;
                    }
                    throw new InvalidDataException($"mod recipe for dtype {Dtype}");
            }
        }
    }
}
