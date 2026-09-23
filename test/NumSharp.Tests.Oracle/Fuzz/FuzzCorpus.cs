using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Fuzz
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

            /// <summary>W14: when true, NumPy raised on this op+operands; the harness asserts
            /// NumSharp ALSO throws (error parity) rather than silently producing a result.</summary>
            public bool Expects_Throw { get; set; }

            /// <summary>
            ///     The exception NumPy raised, captured verbatim at generation time. Cases emitted
            ///     before the error-text upgrade carry none and stay on the weaker "threw
            ///     something" assertion; cases that carry one are held to NumPy's actual message.
            /// </summary>
            public ErrorInfo Error { get; set; }
        }

        /// <summary>NumPy's exception: the Python class name and <c>str(e)</c>, verbatim.</summary>
        public sealed class ErrorInfo
        {
            public string Type { get; set; }
            public string Text { get; set; }
        }

        public sealed class Operand
        {
            public string Dtype { get; set; }
            public long[] Shape { get; set; }
            public long[] Strides { get; set; }
            public long Offset { get; set; }
            public long BufferSize { get; set; }
            public string Buffer { get; set; }

            /// <summary>
            ///     MASKED-ARRAY corpus only: hex of a C-contiguous bool buffer (one byte per element,
            ///     <c>product(Shape)</c> bytes) giving the mask at the DATA view's logical C-order
            ///     positions. Absent (null) means <c>nomask</c> — NumPy's fast path where an operand
            ///     is an ordinary unmasked array. Ordinary (non-ma) tiers never set this, so their
            ///     operand serialization is byte-identical to before.
            /// </summary>
            public string Mask { get; set; }

            /// <summary>
            ///     Explicit writeable flag, serialized by <c>layout_catalog.describe()</c> ONLY when
            ///     the NumPy view was READ-ONLY in a way its strides cannot convey — a SAME-SHAPE
            ///     <c>np.broadcast_to</c> keeps ordinary strides, so without this the reconstructed
            ///     operand was silently writeable and the out_where broadcast-out refusal cells
            ///     (NumPy: ValueError "output array is read-only") could never pass. Absent (null)
            ///     means writeable — every pre-flag corpus row deserializes exactly as before.
            /// </summary>
            public bool? Writeable { get; set; }
        }

        /// <summary>
        ///     One comparable result. The historical shape — a single array described by
        ///     (dtype, shape, C-contiguous bytes) — remains the default; <see cref="Kind"/> widens
        ///     it so ops whose result is NOT one array can be gated by the same corpus.
        ///
        ///     <list type="bullet">
        ///       <item><c>array</c> (or absent) — dtype + shape + bytes, the original contract.</item>
        ///       <item><c>scalar</c> — the same comparison; the op returns a C# scalar that
        ///             <see cref="OpRegistry"/> wraps in a 0-d NDArray (the np.allclose pattern
        ///             already in the registry). Kept distinct only so the corpus reads honestly.</item>
        ///       <item><c>dtype</c> — the op returns a dtype (NPTypeCode); compare
        ///             <see cref="Value"/> against its NumPy dtype name.</item>
        ///       <item><c>text</c> — the op returns a string (printing); compare verbatim.</item>
        ///       <item><c>tuple</c> — the op returns N arrays; compare ARITY and every slot. This
        ///             supersedes the older which/piece params, which gate one slot per case and
        ///             therefore never assert how many slots there were.</item>
        ///     </list>
        /// </summary>
        public sealed class Expected
        {
            /// <summary>array | scalar | dtype | text | tuple. Absent means array (legacy cases).</summary>
            public string Kind { get; set; }

            public string Dtype { get; set; }
            public long[] Shape { get; set; }
            public string Buffer { get; set; }

            /// <summary>
            ///     OPTIONAL correctly-rounded mathematical reference (same dtype/shape/length as
            ///     <see cref="Buffer"/>), carried by the precision tier. Truth is a DIAGNOSTIC
            ///     axis, never a competing pass criterion: a bit-exact-to-NumPy result passes
            ///     without this ever being read ("precise doesn't fail" — byte-parity to NumPy is
            ///     the vision). It is consulted only AFTER a divergence from NumPy is found, to
            ///     adjudicate which side lost precision (MisalignedRegistry branches P1/P2).
            /// </summary>
            public string Truth { get; set; }

            /// <summary>Expected dtype name (kind=dtype) or expected string (kind=text).</summary>
            public string Value { get; set; }

            /// <summary>
            ///     MASKED-ARRAY corpus only (kind=masked / masked_tuple slots): hex of the result's
            ///     <c>getmaskarray</c> — a C-contiguous bool buffer of <c>product(Shape)</c> bytes.
            ///     <see cref="Buffer"/> then holds the <c>filled(0)</c> data (masked slots zeroed), so
            ///     the pair together is the observable masked value: unmasked data bit-for-bit + the
            ///     mask bit-for-bit, with masked-slot underlying data left a wildcard (it is
            ///     implementation-defined for unique/set-ops and merely input-restored for ufuncs).
            /// </summary>
            public string Mask { get; set; }

            /// <summary>Per-slot results for kind=tuple / masked_tuple, in NumPy's tuple order.</summary>
            public Expected[] Slots { get; set; }

            /// <summary>Normalized kind — legacy cases carry none and mean "array".</summary>
            public string KindOrArray => string.IsNullOrEmpty(Kind) ? "array" : Kind;
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

        // -- dtype token <-> NPTypeCode. 13 NumPy-representable types + "char": NumSharp's Char
        // is bit-identical to uint16, so the oracle emits Char cases as uint16-proxy bytes
        // relabelled "uint16"->"char" (gen_oracle._relabel_dtype). Decimal still has no NumPy
        // analog (it rides a separate C#-decimal oracle, not this NumPy differential corpus).
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
            "char" => NPTypeCode.Char,
            "float16" => NPTypeCode.Half,
            "float32" => NPTypeCode.Single,
            "float64" => NPTypeCode.Double,
            "complex128" => NPTypeCode.Complex,
            // Decimal: no NumPy analog — its cases come from the independent C# oracle
            // (gen_decimal_oracle.cs), not the NumPy corpus. SliceFromBytes already handles it.
            "decimal" => NPTypeCode.Decimal,
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
                    return ApplyWriteable(new NDArray(tc, new Shape(o.Shape), false), o);

            var bytes = FromHex(o.Buffer);
            var slice = SliceFromBytes(bytes, tc);                       // Count == bufferSize
            var baseShape = new Shape(new[] { o.BufferSize });          // 1-D contiguous, size == Count
            var storage = new UnmanagedStorage(slice, baseShape);
            var viewShape = new Shape(o.Shape, o.Strides, o.Offset, o.BufferSize); // operand view (alias, no checks)
            return ApplyWriteable(new NDArray(storage, viewShape), o);
        }

        /// <summary>
        ///     Honor the operand's explicit <see cref="Operand.Writeable"/> flag: a same-shape
        ///     broadcast view is read-only in NumPy while its (shape, strides) reconstruction looks
        ///     writeable, so the flag is cleared through the public <c>setflags</c> route (the same
        ///     path the flags oracle audits). Stride-derived read-onlyness (a genuine stride-0
        ///     broadcast dim) needs no flag and is untouched.
        /// </summary>
        /// <param name="nd">The freshly reconstructed operand.</param>
        /// <param name="o">Its corpus descriptor (carries the optional flag).</param>
        /// <returns><paramref name="nd"/>, non-writeable when the descriptor says so.</returns>
        private static NDArray ApplyWriteable(NDArray nd, Operand o)
        {
            if (o.Writeable == false)
                nd.setflags(write: false);
            return nd;
        }

        /// <summary>
        ///     Rebuild the C-contiguous bool mask NDArray a masked operand carries, from its hex and
        ///     the DATA view's logical shape. The bytes are one-per-element in C-order, so the mask is
        ///     a plain contiguous array (never strided) even when the data view is — exactly how NumPy
        ///     stores <c>MaskedArray._mask</c> and how <c>np.ma.array(view, mask=cmask)</c> pairs them.
        /// </summary>
        /// <param name="shape">The data view's shape (the mask has the same shape).</param>
        /// <param name="maskHex">Hex of the C-contiguous bool buffer, or null for nomask.</param>
        /// <returns>The bool mask NDArray, or null when <paramref name="maskHex"/> is null (nomask).</returns>
        public static NDArray ReconstructMask(long[] shape, string maskHex)
        {
            if (maskHex == null)
                return null;                                     // nomask: keep _mask == null (the fast path)
            for (int i = 0; i < shape.Length; i++)
                if (shape[i] == 0)
                    return new NDArray(NPTypeCode.Boolean, new Shape(shape), false);   // empty: no bytes
            var bytes = FromHex(maskHex);
            var slice = ArraySlice.FromBuffer<bool>(bytes, true);
            return new NDArray(new UnmanagedStorage(slice, new Shape(shape)), new Shape(shape));
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
