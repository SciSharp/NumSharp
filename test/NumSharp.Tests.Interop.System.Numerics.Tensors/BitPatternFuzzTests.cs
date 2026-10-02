using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Random bit-pattern fuzzing: the bridge's contract is a raw byte copy (or no copy at all), so ANY bit pattern
    ///     must survive — not only the special values the fidelity tests pick, but bool bytes other than 0/1, NaN
    ///     payloads of every sign and quiet/signalling kind, float16 subnormals, decimal scales and reserved bits no
    ///     arithmetic would produce, and every complex combination. Seeded random bytes are reinterpreted as each of the
    ///     15 element types and pushed through every path, contiguous and strided — the strided export densifies through
    ///     NumSharp's own copy first, which a value-converting kernel could canonicalize.
    /// </summary>
    [TestClass]
    public class BitPatternFuzzTests : TensorsTestBase
    {
        /// <summary>The seed; each dtype derives its own stream from it.</summary>
        private const int Seed = 1009;

        /// <summary>Random arrays generated per dtype.</summary>
        private const int Rounds = 12;

        /// <summary>
        ///     Random bytes as every dtype, through AsTensorSpan, ToTensor and both round trips for a contiguous, a
        ///     transposed, a stepped, a reversed (copy only), a broadcast and a 0-d view, and through ToNDArray /
        ///     AsNDArray / span.ToNDArray for a dense and a sliced tensor — byte-identical every time.
        /// </summary>
        [TestMethod]
        public void RandomBitPatterns_EveryDtype_EveryPath_ByteIdentical()
        {
            var failures = new Failures();
            Dtypes.ForEach(new BitFuzz(failures));
            failures.AssertNone("bit pattern x path");
        }

        /// <summary>Generates <see cref="Rounds"/> random 4×6 arrays of one dtype and checks every path on them.</summary>
        private sealed class BitFuzz : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the fuzzer.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public BitFuzz(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                var rng = new Random(Seed + (int)code * 31);
                for (int round = 0; round < Rounds; round++)
                {
                    // Raw random bytes — Unsafe.SizeOf, not Marshal.SizeOf, which reports 4 for bool.
                    var bytes = new byte[24 * Unsafe.SizeOf<T>()];
                    rng.NextBytes(bytes);
                    T[] values = MemoryMarshal.Cast<byte, T>(bytes).ToArray();
                    NDArray flat = np.array(values);
                    NDArray m = flat.reshape(4, 6);

                    (string Name, NDArray View)[] views =
                    {
                        ("contiguous", m),
                        ("transposed", m.T),
                        ("stepped", m["::2, 1::2"]),
                        ("reversed", m[":, ::-1"]),
                        ("broadcast", np.broadcast_to(m["1"], new Shape(3, 6))),
                        ("0-d", m["2, 3"]),
                    };
                    foreach (var (name, view) in views)
                        CheckExport<T>(_failures, $"{code} round {round} {name}", view);

                    Tensor<T> dense = Tensor.Create(values, new nint[] { 4, 6 });
                    Tensor<T> sliced = dense.Slice(new NRange[] { new NRange(1, 3), new NRange(2, 5) });
                    CheckImport(_failures, $"{code} round {round} dense tensor", dense, code);
                    CheckImport(_failures, $"{code} round {round} sliced tensor", sliced, code);
                }
            }
        }

        /// <summary>Every export path of one view must reproduce NumSharp's own bytes.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The view's name.</param>
        /// <param name="view">The view.</param>
        private static void CheckExport<T>(Failures failures, string cell, NDArray view) where T : unmanaged
        {
            T[] want = Oracle.Elements<T>(view);
            if (!Oracle.HasNegativeStride(view))
                failures.Check($"{cell} / AsTensorSpan", () =>
                {
                    using var h = view.AsTensorSpan<T>();
                    return Oracle.SameBytes(Oracle.Flatten(h.ReadOnlySpan), want) ? null : "bytes differ";
                });
            failures.Check($"{cell} / ToTensor", () => Oracle.SameBytes(Oracle.Flatten(view.ToTensor<T>()), want) ? null : "bytes differ");
            failures.Check($"{cell} / ToTensor->ToNDArray", () =>
            {
                using NDArray back = view.ToTensor<T>().ToNDArray();
                return Oracle.SameBytes(Oracle.Elements<T>(back), want) ? null : "bytes differ";
            });
            failures.Check($"{cell} / ToTensor->AsNDArray", () =>
            {
                using NDArray back = view.ToTensor<T>().AsNDArray();
                return Oracle.SameBytes(Oracle.Elements<T>(back), want) ? null : "bytes differ";
            });
        }

        /// <summary>Every import path of one tensor must reproduce the tensor's own bytes.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="failures">Where failing cells go.</param>
        /// <param name="cell">The tensor's name.</param>
        /// <param name="t">The tensor.</param>
        /// <param name="code">Its NumSharp dtype.</param>
        private static void CheckImport<T>(Failures failures, string cell, Tensor<T> t, NPTypeCode code) where T : unmanaged
        {
            T[] want = Oracle.Flatten(t);
            long[] lengths = Oracle.Longs(t.Lengths);
            failures.Check($"{cell} / ToNDArray", () =>
            {
                using NDArray copy = t.ToNDArray();
                return ExportContract.Imported(copy, code, lengths, want);
            });
            failures.Check($"{cell} / AsNDArray", () =>
            {
                using NDArray view = t.AsNDArray();
                return ExportContract.Imported(view, code, lengths, want);
            });
            failures.Check($"{cell} / span.ToNDArray", () =>
            {
                using NDArray copy = t.AsReadOnlyTensorSpan().ToNDArray();
                return ExportContract.Imported(copy, code, lengths, want);
            });
        }
    }
}
