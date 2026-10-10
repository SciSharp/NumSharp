using System;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Bit-pattern fuzzing: seeded random BYTES reinterpreted as each dtype — bool bytes other than 0 and 1, every NaN
    ///     payload (signalling ones included), subnormals, negative zeros, extreme integers — must cross every path
    ///     byte-identical: both export verbs, both dense-tensor verbs, both import verbs, the round trips, and ORT's own
    ///     <c>Identity</c> kernel, from contiguous and strided sources. A conversion anywhere (a canonicalized NaN, a bool
    ///     normalized to 1, a quieted signalling NaN) shows up as a byte difference. Decimal is converted by design and is
    ///     fuzzed over valid decimals against NumSharp's own conversion; Complex has no ONNX type.
    /// </summary>
    [TestClass]
    public class BitPatternFuzzTests : OnnxTestBase
    {
        /// <summary>The seed the random bytes start from.</summary>
        private const int Seed = 777;

        /// <summary>Random arrays per dtype.</summary>
        private const int Arrays = 40;

        /// <summary>Random bit patterns cross every path byte-identical.</summary>
        [TestMethod]
        public void RandomBitPatterns_CrossEveryPath_ByteIdentical()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Fuzz(failures));
            failures.AssertNone("bit pattern x path");
        }

        /// <summary>The fuzz's generic body, run once per dtype.</summary>
        private sealed class Fuzz : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the fuzz.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal Fuzz(Failures failures) => _failures = failures;

            /// <summary>Fuzzes one dtype (Complex excluded).</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator (unused: the values are random bytes).</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                switch (code)
                {
                    case NPTypeCode.Complex:
                        return;
                    case NPTypeCode.Char:
                        Paths<T, ushort>(code);
                        return;
                    case NPTypeCode.Half:
                        Paths<T, Float16>(code);
                        return;
                    case NPTypeCode.Decimal:
                        Paths<T, double>(code);
                        return;
                    default:
                        Paths<T, T>(code);
                        return;
                }
            }

            /// <summary>Builds random arrays of one dtype and checks every path.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <typeparam name="X">The element type ORT sees.</typeparam>
            /// <param name="code">The dtype.</param>
            private unsafe void Paths<T, X>(NPTypeCode code) where T : unmanaged where X : unmanaged
            {
                var rng = new Random(Seed ^ (int)code);
                using InferenceSession session = OpenSession(IdentityModel(code));
                for (int i = 0; i < Arrays; i++)
                {
                    int n = rng.Next(1, 64);
                    var nd = new NDArray(code, new Shape(n), fillZeros: false);
                    if (code == NPTypeCode.Decimal)
                    {
                        // Random VALID decimals: random 96-bit mantissas at every scale (random bytes are mostly invalid decimals).
                        var p = (decimal*)nd.Storage.Address;
                        for (int k = 0; k < n; k++)
                            p[k] = new decimal(rng.Next(), rng.Next(), rng.Next(), rng.Next(2) == 0, (byte)rng.Next(29));
                    }
                    else
                    {
                        var bytes = new byte[n * sizeof(T)];
                        rng.NextBytes(bytes);
                        Marshal.Copy(bytes, 0, (IntPtr)nd.Storage.Address, bytes.Length);
                    }

                    foreach (string layout in new[] { "contiguous", "reversed", "stepped" })
                    {
                        NDArray view = layout switch
                        {
                            "contiguous" => nd,
                            "reversed" => nd["::-1"],
                            _ => nd["::2"],
                        };
                        // What crosses: the raw elements, or for Decimal NumSharp's own float64 conversion of them.
                        byte[] want;
                        if (code == NPTypeCode.Decimal)
                        {
                            using NDArray conversion = view.astype(NPTypeCode.Double);
                            want = Oracle.LogicalBytes(conversion);
                        }
                        else
                        {
                            want = Oracle.LogicalBytes(view);
                        }
                        long[] dims = view.shape;
                        NPTypeCode back = Oracle.CrossedCode(code);
                        string cell = $"{code} #{i} n={n} {layout}";

                        _failures.Check($"{cell} / ToOrtValue->ToNDArray", () =>
                        {
                            using OrtValue v = view.ToOrtValue();
                            using NDArray b = v.ToNDArray();
                            return ExportContract.Imported(b, back, dims, want);
                        });
                        // Every layout: a view ORT cannot read in place crosses as a copy under the default mode.
                        _failures.Check($"{cell} / AsOrtValue->AsNDArray", () =>
                        {
                            using OrtTensor h = view.AsOrtValue();
                            using NDArray b = h.Value.AsNDArray();
                            return ExportContract.Imported(b, back, dims, want);
                        });
                        _failures.Check($"{cell} / ToDenseTensor->ToNDArray", () =>
                        {
                            using NDArray b = view.ToDenseTensor<X>().ToNDArray();
                            return ExportContract.Imported(b, back, dims, want);
                        });
                        _failures.Check($"{cell} / AsDenseTensor->AsNDArray", () =>
                        {
                            using OrtTensor<X> h = view.AsDenseTensor<X>();
                            using NDArray b = h.Tensor.AsNDArray();
                            return ExportContract.Imported(b, back, dims, want);
                        });
                        _failures.Check($"{cell} / ORT Identity", () =>
                        {
                            using NDArray b = session.Run(view, casting: Oracle.RunCasting(code));
                            return ExportContract.Imported(b, back, dims, want);
                        });
                    }
                }
            }
        }
    }
}
