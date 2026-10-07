using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Tests.Fuzz;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The committed NumPy 2.4.2 oracle for the fixed-width string dtypes (<c>Strings/corpus/strings.jsonl</c>, written by
    ///     <c>test/oracle/gen_strings_oracle.py</c>): its case model, the exact reconstruction of every operand view from
    ///     its raw bytes, and the decoding of the text values the generator stores as hex.
    /// </summary>
    /// <remarks>
    ///     Text never travels as a JSON string: a Python <c>str</c> is <c>{"u": UTF-32LE hex}</c> and a <c>bytes</c> is
    ///     <c>{"b": hex}</c>, so NULs, lone surrogates and supplementary-plane characters survive the round trip
    ///     unchanged (System.Text.Json refuses lone surrogates in string literals).
    /// </remarks>
    internal static class StringOracleCorpus
    {
        private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

        /// <summary>One recorded NumPy call (see the generator's module docstring for the field meanings).</summary>
        internal sealed class Case
        {
            /// <summary>Unique id: <c>section/op/n</c> (or a descriptive id in the dtype section).</summary>
            public string Id { get; set; }

            /// <summary>The op key <see cref="StringOracleOps.Invoke"/> dispatches on.</summary>
            public string Op { get; set; }

            /// <summary>The call's arguments other than the operands.</summary>
            public Dictionary<string, JsonElement> Params { get; set; }

            /// <summary>The arrays the call reads, in call order.</summary>
            public Operand[] Operands { get; set; }

            /// <summary>1: NumSharp must match NumPy now. 2: NumPy computes on text, NumSharp may defer.</summary>
            public int Phase { get; set; }

            /// <summary>The operand layout name(s) (diagnostics only).</summary>
            public string Layout { get; set; }

            /// <summary>The expectation when NumPy returned (kind-dependent object); undefined when it raised.</summary>
            public JsonElement Expected { get; set; }

            /// <summary>NumPy's exception when it raised, else null.</summary>
            public ErrorInfo Error { get; set; }
        }

        /// <summary>NumPy's exception: its first well-known class and <c>str(e)</c>, verbatim.</summary>
        internal sealed class ErrorInfo
        {
            /// <summary>The Python class name (UFuncTypeError is recorded as TypeError).</summary>
            public string Type { get; set; }

            /// <summary>The message.</summary>
            public string Text { get; set; }
        }

        /// <summary>An operand: a view (shape, element strides, element offset) over a 1-D base buffer.</summary>
        internal sealed class Operand
        {
            /// <summary>The descriptor's <c>str</c> (<c>&lt;U3</c>, <c>|S5</c>, <c>|b1</c>, <c>&lt;i8</c>).</summary>
            public string Dtype { get; set; }

            /// <summary>The view's dimensions.</summary>
            public long[] Shape { get; set; }

            /// <summary>The view's strides in elements (negative and zero strides included).</summary>
            public long[] Strides { get; set; }

            /// <summary>The view's first element, in elements from the base start.</summary>
            public long Offset { get; set; }

            /// <summary>The base buffer's length in elements.</summary>
            public long BufferSize { get; set; }

            /// <summary>The base buffer's raw bytes as hex.</summary>
            public string Buffer { get; set; }

            /// <summary>False when NumPy's view was read-only beyond what its strides show; null otherwise.</summary>
            public bool? Writeable { get; set; }
        }

        /// <summary>The corpus path next to the test assembly (copied there by the csproj rule).</summary>
        internal static string CorpusPath =>
            Path.Combine(AppContext.BaseDirectory, "Strings", "corpus", "strings.jsonl");

        /// <summary>Reads every case of the corpus.</summary>
        /// <returns>The cases in file order.</returns>
        /// <exception cref="FileNotFoundException">The corpus was not copied next to the test assembly.</exception>
        internal static List<Case> Load()
        {
            var cases = new List<Case>();
            foreach (var line in File.ReadLines(CorpusPath, Encoding.ASCII))
                if (line.Length > 0)
                    cases.Add(JsonSerializer.Deserialize<Case>(line, Json));
            return cases;
        }

        /// <summary>
        ///     Rebuilds the exact operand NumPy saw: the base buffer becomes a 1-D storage of its dtype that owns its data,
        ///     and the operand is either that base itself (when NumPy's operand WAS the base: the whole buffer, stride 1,
        ///     offset 0 — so <c>flags.owndata</c> is true as in NumPy) or a non-owning view of it with the recorded shape,
        ///     strides and offset (so a reversed, strided, transposed, broadcast, empty or 0-d view is that view, not a
        ///     copy).
        /// </summary>
        /// <param name="o">The operand descriptor.</param>
        /// <returns>The base or the view.</returns>
        /// <exception cref="NotSupportedException">A numeric dtype the general corpus has no mapping for.</exception>
        internal static NDArray Reconstruct(Operand o)
        {
            var descr = np.dtype(o.Dtype);
            bool isString = descr.Meta is StringDTypeMeta;
            if (!isString)
            {
                // Numeric operands (masks, indices, numbers) ride the general corpus' reconstruction by NumPy dtype name.
                return FuzzCorpus.Reconstruct(new FuzzCorpus.Operand
                {
                    Dtype = descr.name,
                    Shape = o.Shape,
                    Strides = o.Strides,
                    Offset = o.Offset,
                    BufferSize = o.BufferSize,
                    Buffer = o.Buffer,
                    Writeable = o.Writeable,
                });
            }

            NDArray baseArray;
            if (o.BufferSize == 0)
            {
                baseArray = np.zeros(new Shape(0L), descr);
            }
            else
            {
                var bytes = Convert.FromHexString(o.Buffer);
                var slice = new StringArraySlice(ArraySlice.FromBuffer<byte>(bytes, true), descr);
                baseArray = new NDArray(new UnmanagedStorage(slice, new Shape(new[] { o.BufferSize })));
            }

            bool isBase = o.Shape.Length == 1 && o.Shape[0] == o.BufferSize && o.Offset == 0
                          && (o.BufferSize <= 1 || o.Strides[0] == 1);
            NDArray view = isBase
                ? baseArray
                : new NDArray(baseArray.Storage.Alias(new Shape(o.Shape, o.Strides, o.Offset, o.BufferSize)));
            if (o.Writeable == false)
                view.setflags(write: false);
            return view;
        }

        /// <summary>
        ///     A value the generator encoded with <c>enc()</c>: <c>{"u"}</c> → <see cref="string"/>, <c>{"b"}</c> →
        ///     <c>byte[]</c>, <c>{"none"}</c> → null, <c>{"bool"}</c> / <c>{"int"}</c> / <c>{"float"}</c> → bool / long /
        ///     double, a JSON array → a <see cref="List{T}"/> of decoded items.
        /// </summary>
        /// <param name="e">The encoded value.</param>
        /// <returns>The decoded value.</returns>
        /// <exception cref="FormatException">An unknown encoding.</exception>
        internal static object Decode(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Array)
                return e.EnumerateArray().Select(Decode).ToList();
            if (e.ValueKind == JsonValueKind.Null)
                return null;
            if (e.TryGetProperty("u", out var u))
                return Utf32HexToString(u.GetString());
            if (e.TryGetProperty("b", out var b))
                return Convert.FromHexString(b.GetString());
            if (e.TryGetProperty("none", out _))
                return null;
            if (e.TryGetProperty("bool", out var bo))
                return bo.GetBoolean();
            if (e.TryGetProperty("int", out var i))
                return i.GetInt64();
            if (e.TryGetProperty("float", out var f))
                return f.GetDouble();
            throw new FormatException($"unknown value encoding {e}");
        }

        /// <summary>
        ///     A Python <c>str</c> from its UTF-32LE bytes, mapped to .NET the way NumSharp's <c>U</c> decoding maps it: a
        ///     supplementary code point becomes a surrogate pair, a lone surrogate code point stays one UTF-16 unit.
        /// </summary>
        /// <param name="hex">The UTF-32LE bytes as hex.</param>
        /// <returns>The .NET string.</returns>
        internal static string Utf32HexToString(string hex)
        {
            var bytes = Convert.FromHexString(hex);
            var sb = new StringBuilder(bytes.Length / 4);
            for (int i = 0; i + 3 < bytes.Length; i += 4)
            {
                int cp = BitConverter.ToInt32(bytes, i);
                if (cp > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32(cp));
                else
                    sb.Append((char)cp);
            }
            return sb.ToString();
        }

        /// <summary>
        ///     The C-order bytes of an array of any layout — the bytes NumPy's <c>np.ascontiguousarray(r).tobytes()</c>
        ///     gives, so a result is compared element for element in logical order whatever its strides.
        /// </summary>
        /// <param name="r">The array (any dtype NumSharp stores).</param>
        /// <returns>size × itemsize bytes.</returns>
        internal static unsafe byte[] CBytes(NDArray r)
        {
            if (r.size == 0)
                return Array.Empty<byte>();
            // A copy in C order is the logical element sequence; a C-contiguous non-broadcast array already is one.
            var c = r.Shape.IsContiguous && !r.Shape.IsBroadcasted ? r : r.copy('C');
            int isz = c.dtype.itemsize;
            long n = c.size * isz;
            var dst = new byte[n];
            var src = (byte*)c.Storage.Address + c.Shape.offset * isz;
            new ReadOnlySpan<byte>(src, checked((int)n)).CopyTo(dst);
            return dst;
        }

        /// <summary>
        ///     The whole base buffer under an operand view — after an in-place call, the bytes inside AND outside the
        ///     view's window, so a write that strays outside the selected elements is caught.
        /// </summary>
        /// <param name="view">An operand built by <see cref="Reconstruct"/> (it shares its base's storage).</param>
        /// <returns>The base buffer's bytes.</returns>
        internal static unsafe byte[] BaseBytes(NDArray view)
        {
            var arr = view.Storage.InternalArray;
            long n = arr is StringArraySlice s ? s.BytesLength : arr.Count * arr.ItemLength;
            var dst = new byte[n];
            new ReadOnlySpan<byte>(arr.Address, checked((int)n)).CopyTo(dst);
            return dst;
        }
    }
}
