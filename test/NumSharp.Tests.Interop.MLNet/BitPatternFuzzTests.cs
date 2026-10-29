using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Bit-pattern fuzzing: seeded random BYTES reinterpreted as each dtype — bool bytes other than 0 and 1, every NaN
    ///     payload (signalling ones included), subnormals, negative zeros, extreme integers — must cross every path
    ///     byte-identical: both data-view shapes read through ML.NET's cursors, the snapshot, <c>ToVBuffer</c>, and the
    ///     imports back, from contiguous and strided sources. Half and Decimal are converted by design: they must cross as
    ///     exactly NumSharp's own conversion (NumPy's rules: a signalling NaN stays signalling); Complex has no ML.NET type.
    /// </summary>
    [TestClass]
    public class BitPatternFuzzTests : MLNetTestBase
    {
        /// <summary>The seed the random bytes start from.</summary>
        private const int Seed = 777;

        /// <summary>Random arrays per dtype.</summary>
        private const int Arrays = 40;

        /// <summary>Random bit patterns cross every path byte-identical (or as NumSharp's own conversion).</summary>
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
                        Paths<T, float>(code);
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
            /// <typeparam name="X">The element type ML.NET sees.</typeparam>
            /// <param name="code">The dtype.</param>
            private unsafe void Paths<T, X>(NPTypeCode code) where T : unmanaged where X : unmanaged
            {
                var rng = new Random(Seed ^ (int)code);
                NPTypeCode back = Oracle.CrossedCode(code);
                for (int i = 0; i < Arrays; i++)
                {
                    int rows = rng.Next(1, 9), cols = rng.Next(1, 7);
                    var nd = new NDArray(code, new Shape(rows, cols), fillZeros: false);
                    if (code == NPTypeCode.Decimal)
                    {
                        var p = (decimal*)nd.Storage.Address;
                        for (int k = 0; k < rows * cols; k++)
                            p[k] = new decimal(rng.Next(), rng.Next(), rng.Next(), rng.Next(2) == 0, (byte)rng.Next(29));
                    }
                    else
                    {
                        var bytes = new byte[rows * cols * sizeof(T)];
                        rng.NextBytes(bytes);
                        Marshal.Copy(bytes, 0, (IntPtr)nd.Storage.Address, bytes.Length);
                    }

                    foreach (string layout in new[] { "contiguous", "reversed", "transposed" })
                    {
                        NDArray view = layout switch
                        {
                            "contiguous" => nd,
                            "reversed" => nd["::-1, ::-1"],
                            _ => nd.T,
                        };
                        byte[] want;
                        if (back != code && code != NPTypeCode.Char)
                        {
                            using NDArray conversion = view.astype(back);
                            want = Oracle.LogicalBytes(conversion);
                        }
                        else
                        {
                            want = Oracle.LogicalBytes(view);
                        }
                        long[] dims = view.shape;
                        string[] names = ExportContract.Names(view);
                        string cell = $"{code} #{i} {Oracle.Describe(dims)} {layout}";

                        _failures.Check($"{cell} / AsDataView(column) via cursor + import", () =>
                        {
                            using NDArrayDataView dv = view.AsDataView("F");
                            byte[] read = Oracle.Bytes(Oracle.VectorRows<X>(dv, "F").SelectMany(r => r).ToArray());
                            if (!Oracle.SameBytes(read, want))
                                return "rows read through the cursor differ";
                            using NDArray b = dv.ToNDArray("F");
                            return ExportContract.Imported(b, back, dims, want);
                        });
                        _failures.Check($"{cell} / ToDataView(columns) via cursor", () =>
                        {
                            using NDArrayDataView dv = view.ToDataView(names);
                            byte[] read = Oracle.Bytes(Oracle.ScalarRows<X>(dv, names).SelectMany(r => r).ToArray());
                            return Oracle.SameBytes(read, want) ? null : "columns read through the cursor differ";
                        });
                        _failures.Check($"{cell} / ToVBuffer->ToNDArray+AsNDArray", () =>
                        {
                            VBuffer<X> buffer = view.ToVBuffer<X>();
                            if (!Oracle.SameBytes(MemoryMarshal.AsBytes(buffer.GetValues()), want))
                                return "buffer values differ";
                            using NDArray copy = buffer.ToNDArray();
                            using NDArray shared = buffer.AsNDArray();
                            long[] flat = { want.Length / sizeof(X) };
                            return ExportContract.Imported(copy, back, flat, want) ?? ExportContract.Imported(shared, back, flat, want);
                        });
                    }
                }
            }
        }
    }
}
