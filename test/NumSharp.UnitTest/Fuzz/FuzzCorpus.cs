using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Reads the committed NumPy oracle corpus (JSONL) and reconstructs the EXACT operand
    ///     views the cases describe — including broadcast (stride-0), negative strides, and
    ///     offset slices — directly from raw bytes, so no NumPy is needed at test time.
    ///
    ///     Operand layout is described by (dtype, shape, element-strides, element-offset,
    ///     bufferSize, base-buffer-hex). We reconstruct by wrapping the base buffer in a
    ///     contiguous storage (size == bufferSize) and aliasing it with the operand's view shape.
    /// </summary>
    public static class FuzzCorpus
    {
        private static readonly JsonSerializerOptions J = new() { PropertyNameCaseInsensitive = true };

        public sealed class Case
        {
            public string Id { get; set; }
            public string Op { get; set; }
            public Dictionary<string, JsonElement> Params { get; set; }
            public Operand[] Operands { get; set; }
            public Expected Expected { get; set; }
            public string Layout { get; set; }
            public string Valueclass { get; set; }

            /// <summary>W11: when true, a single stored operand is passed to a binary op as BOTH
            /// arguments via the SAME reference (true input aliasing: a op a).</summary>
            public bool Alias { get; set; }
        }

        public sealed class Operand
        {
            public string Dtype { get; set; }
            public long[] Shape { get; set; }
            public long[] Strides { get; set; }
            public long Offset { get; set; }
            public long BufferSize { get; set; }
            public string Buffer { get; set; }
        }

        public sealed class Expected
        {
            public string Dtype { get; set; }
            public long[] Shape { get; set; }
            public string Buffer { get; set; }
        }

        /// <summary>Resolve a corpus file copied next to the test assembly under Fuzz/corpus/.</summary>
        public static string CorpusPath(string fileName)
            => Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus", fileName);

        public static List<Case> Load(string fileName)
        {
            var path = CorpusPath(fileName);
            var list = new List<Case>();
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                list.Add(JsonSerializer.Deserialize<Case>(line, J));
            }
            return list;
        }

        // -- dtype token <-> NPTypeCode (13 NumPy-representable types; Char/Decimal have no NumPy analog) --
        public static NPTypeCode DtypeToTC(string name) => name switch
        {
            "bool" => NPTypeCode.Boolean,
            "int8" => NPTypeCode.SByte,
            "uint8" => NPTypeCode.Byte,
            "int16" => NPTypeCode.Int16,
            "uint16" => NPTypeCode.UInt16,
            "int32" => NPTypeCode.Int32,
            "uint32" => NPTypeCode.UInt32,
            "int64" => NPTypeCode.Int64,
            "uint64" => NPTypeCode.UInt64,
            "float16" => NPTypeCode.Half,
            "float32" => NPTypeCode.Single,
            "float64" => NPTypeCode.Double,
            "complex128" => NPTypeCode.Complex,
            _ => throw new NotSupportedException($"dtype '{name}' has no NumSharp NPTypeCode mapping")
        };

        public static byte[] FromHex(string h)
            => string.IsNullOrEmpty(h) ? Array.Empty<byte>() : Convert.FromHexString(h);

        private static IArraySlice SliceFromBytes(byte[] bytes, NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => ArraySlice.FromBuffer<bool>(bytes, true),
            NPTypeCode.SByte => ArraySlice.FromBuffer<sbyte>(bytes, true),
            NPTypeCode.Byte => ArraySlice.FromBuffer<byte>(bytes, true),
            NPTypeCode.Int16 => ArraySlice.FromBuffer<short>(bytes, true),
            NPTypeCode.UInt16 => ArraySlice.FromBuffer<ushort>(bytes, true),
            NPTypeCode.Int32 => ArraySlice.FromBuffer<int>(bytes, true),
            NPTypeCode.UInt32 => ArraySlice.FromBuffer<uint>(bytes, true),
            NPTypeCode.Int64 => ArraySlice.FromBuffer<long>(bytes, true),
            NPTypeCode.UInt64 => ArraySlice.FromBuffer<ulong>(bytes, true),
            NPTypeCode.Char => ArraySlice.FromBuffer<char>(bytes, true),
            NPTypeCode.Half => ArraySlice.FromBuffer<Half>(bytes, true),
            NPTypeCode.Single => ArraySlice.FromBuffer<float>(bytes, true),
            NPTypeCode.Double => ArraySlice.FromBuffer<double>(bytes, true),
            NPTypeCode.Decimal => ArraySlice.FromBuffer<decimal>(bytes, true),
            NPTypeCode.Complex => ArraySlice.FromBuffer<System.Numerics.Complex>(bytes, true),
            _ => throw new NotSupportedException($"NPTypeCode {tc} unsupported in SliceFromBytes")
        };

        /// <summary>Reconstruct the exact operand NDArray the corpus case describes.</summary>
        public static NDArray Reconstruct(Operand o)
        {
            var tc = DtypeToTC(o.Dtype);

            // Empty operands: strides/offset are vacuous (0 elements). Build a plain empty array.
            for (int i = 0; i < o.Shape.Length; i++)
                if (o.Shape[i] == 0)
                    return new NDArray(tc, new Shape(o.Shape), false);

            var bytes = FromHex(o.Buffer);
            var slice = SliceFromBytes(bytes, tc);                       // Count == bufferSize
            var baseShape = new Shape(new[] { o.BufferSize });          // 1-D contiguous, size == Count
            var storage = new UnmanagedStorage(slice, baseShape);
            var viewShape = new Shape(o.Shape, o.Strides, o.Offset, o.BufferSize); // operand view (alias, no checks)
            return new NDArray(storage, viewShape);
        }

        /// <summary>Materialize an op result to C-contiguous, offset-0 logical bytes for bit comparison.</summary>
        public static unsafe byte[] ResultBytes(NDArray r)
        {
            int isz = r.typecode.SizeOf();
            long n = r.size;
            var outb = new byte[checked((int)(n * isz))];
            if (n == 0)
                return outb;

            var c = r;
            if (!(c.Shape.IsContiguous && c.Shape.offset == 0))
                c = np.ascontiguousarray(r);
            if (!(c.Shape.IsContiguous && c.Shape.offset == 0))
                c = c.copy();

            byte* src = (byte*)c.Address;
            fixed (byte* dstp = outb)
                Buffer.MemoryCopy(src, dstp, outb.Length, outb.Length);
            return outb;
        }
    }
}
