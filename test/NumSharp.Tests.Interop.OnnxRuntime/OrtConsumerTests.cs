using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Consumer-driven coverage: ONNX Runtime itself — real sessions running real models — reads what the bridge hands
    ///     it and writes where the bridge points it. Every dtype × catalogued layout is fed to an identity model through
    ///     every feed route the package offers (the <c>session.Run(NDArray)</c> tier, raw <see cref="OrtValue"/>s from both
    ///     export verbs, the legacy <see cref="NamedOnnxValue"/> surface with both <see cref="DenseTensor{T}"/> verbs, and
    ///     zero-copy outputs), and ORT's answer must equal NumSharp's own elements; pre-allocated outputs must receive
    ///     exactly their window; and real arithmetic (<c>Add</c> with broadcasting, <c>MatMul</c>) over strided and broadcast
    ///     inputs must equal NumSharp's own result.
    /// </summary>
    [TestClass]
    public class OrtConsumerTests : OnnxTestBase
    {
        /// <summary>Every dtype × layout × feed route through a real identity model returns NumSharp's elements.</summary>
        [TestMethod]
        public void EveryDtype_EveryLayout_EveryFeedRoute_IdentityReturnsTheElements()
        {
            var failures = new Failures();
            Dtypes.ForEach(new IdentitySweep(failures));
            failures.AssertNone("dtype x layout x feed route");
        }

        /// <summary>
        ///     A pre-allocated output that is a window of a larger array (a contiguous slice at an offset) receives the result,
        ///     and ORT writes nothing outside it — the bridge shares exactly the window's bytes, not the whole buffer.
        /// </summary>
        [TestMethod]
        public void PreallocatedOutputWindow_ReceivesTheResult_AndNothingOutsideItChanges()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3, 4);
            using NDArray big = np.full(new Shape(10, 4), -1f);
            using NDArray window = big["2:5"];
            session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", window } });

            Assert.IsTrue(np.array_equal(window, x), "the window holds the result");
            using NDArray before = big["0:2"];
            using NDArray after = big["5:"];
            Assert.IsTrue(np.all(before == -1f), "rows before the window were written");
            Assert.IsTrue(np.all(after == -1f), "rows after the window were written");

            // A 1-D window at an odd offset: element-exact edges.
            using NDArray flatBig = np.full(new Shape(20), -1f);
            using NDArray flatWindow = flatBig["3:15"];
            using NDArray flatX = x.flatten();
            session.RunInto(new Dictionary<string, NDArray> { { "X", flatX } }, new Dictionary<string, NDArray> { { "Y", flatWindow } });
            Assert.IsTrue(np.array_equal(flatWindow, flatX), "the 1-D window holds the result");
            Assert.AreEqual(-1f, flatBig.GetSingle(2), "the element before the window");
            Assert.AreEqual(-1f, flatBig.GetSingle(15), "the element after the window");
        }

        /// <summary>
        ///     ORT's own arithmetic over strided, transposed, reversed and broadcast inputs (each fed through the bridge, which
        ///     copies what ORT cannot share) equals NumSharp's own: <c>Add</c> with ORT-side broadcasting, and <c>MatMul</c>.
        ///     The values are small integers, exact in float32, so equality is exact.
        /// </summary>
        [TestMethod]
        public void AddAndMatMul_OverStridedAndBroadcastInputs_MatchNumSharp()
        {
            using InferenceSession add = OpenSession("add_f32");
            using InferenceSession matmul = OpenSession("matmul_f32");
            using NDArray m = Arange(NPTypeCode.Single, 4, 6);
            using NDArray row = Arange(NPTypeCode.Single, 6);
            var addCases = new (string Name, NDArray A, NDArray B)[]
            {
                ("contiguous + broadcast row", m, row),
                ("stepped rows + reversed row", m["::2"], row["::-1"]),
                ("transposed + column", m.T, Arange(NPTypeCode.Single, 6, 1)),
                ("broadcast view + transposed slice", np.broadcast_to(row, new Shape(3, 6)), m["1:4"]),
            };
            foreach (var (name, a, b) in addCases)
            {
                IReadOnlyDictionary<string, NDArray> outputs = add.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
                using NDArray c = outputs["C"];
                using NDArray want = np.add(a, b);
                Assert.IsTrue(np.array_equal(c, want), $"Add: {name}");
            }

            var matmulCases = new (string Name, NDArray A, NDArray B)[]
            {
                ("contiguous", m, Arange(NPTypeCode.Single, 6, 3)),
                ("transposed @ transposed", m.T, Arange(NPTypeCode.Single, 5, 4).T),
                ("stepped @ reversed", m[":, ::2"], Arange(NPTypeCode.Single, 3, 2)["::-1"]),
            };
            foreach (var (name, a, b) in matmulCases)
            {
                IReadOnlyDictionary<string, NDArray> outputs = matmul.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
                using NDArray c = outputs["C"];
                using NDArray want = np.matmul(a, b);
                Assert.IsTrue(np.array_equal(c, want), $"MatMul: {name}");
            }
        }

        /// <summary>The identity sweep's generic body, run once per dtype with the dtype's identity model.</summary>
        private sealed class IdentitySweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal IdentitySweep(Failures failures) => _failures = failures;

            /// <summary>Runs every layout of one dtype through every feed route (Complex has no ONNX type: skipped).</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                switch (code)
                {
                    case NPTypeCode.Complex:
                        return;
                    case NPTypeCode.Char:
                        Routes<T, ushort>(code, value);
                        return;
                    case NPTypeCode.Half:
                        Routes<T, Float16>(code, value);
                        return;
                    case NPTypeCode.Decimal:
                        Routes<T, double>(code, value);
                        return;
                    default:
                        Routes<T, T>(code, value);
                        return;
                }
            }

            /// <summary>Feeds every layout of one dtype through every route and checks ORT's identity output.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <typeparam name="X">The element type ORT sees.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            private void Routes<T, X>(NPTypeCode code, Func<long, T> value) where T : unmanaged where X : unmanaged
            {
                using InferenceSession session = OpenSession(IdentityModel(code));
                using var runOptions = new RunOptions();
                NPTypeCode backCode = Oracle.CrossedCode(code);
                foreach (var (name, build) in Layouts.All(value))
                {
                    NDArray view = build();
                    T[] want = Oracle.Elements<T>(view);
                    X[] crossed = Oracle.Cross<T, X>(want);
                    byte[] bytes = Oracle.Bytes(crossed);
                    long[] dims = view.shape;
                    // Can ORT read the view's own buffer? Then AsOrtValue / AsDenseTensor share it, else they hand ORT a
                    // copy (the default mode). Every layout is fed either way; the flag only says which one a handle must be.
                    bool shareable = code != NPTypeCode.Decimal && view.Shape.IsContiguous;
                    string cell = $"{code} / {name}";

                    _failures.Check($"{cell} / session.Run(NDArray)", () =>
                    {
                        using NDArray y = session.Run(view, casting: Oracle.RunCasting(code));
                        return ExportContract.Imported(y, backCode, dims, bytes);
                    });

                    _failures.Check($"{cell} / raw Run(ToOrtValue)", () =>
                    {
                        using OrtValue x = view.ToOrtValue();
                        return RawRun(session, runOptions, x, backCode, dims, bytes);
                    });

                    _failures.Check($"{cell} / raw Run(AsOrtValue)", () =>
                    {
                        using OrtTensor h = view.AsOrtValue();
                        return h.IsView != shareable
                            ? $"IsView {h.IsView}, expected {shareable}"
                            : RawRun(session, runOptions, h.Value, backCode, dims, bytes);
                    });

                    _failures.Check($"{cell} / NamedOnnxValue(ToDenseTensor)", () =>
                        LegacyRun(session, NamedOnnxValue.CreateFromTensor("X", view.ToDenseTensor<X>()), backCode, dims, bytes));

                    // Every layout: the tensor is always row-major (a view ORT cannot read in place crosses as a C-order
                    // copy), so ORT's NamedOnnxValue route, which refuses a column-major tensor, accepts all of them. This
                    // was gap G1 of docs/plans/onnxruntime-gap-review.md: a Fortran-ordered array could not be fed at all.
                    _failures.Check($"{cell} / NamedOnnxValue(AsDenseTensor)", () =>
                    {
                        using OrtTensor<X> h = view.AsDenseTensor<X>();
                        return h.IsView != shareable
                            ? $"IsView {h.IsView}, expected {shareable}"
                            : LegacyRun(session, NamedOnnxValue.CreateFromTensor("X", h.Tensor), backCode, dims, bytes);
                    });

                    // The tensor's Memory as an OrtValue (the route the README names for a DenseTensor handle): it read a
                    // column-major share at the wrong indices, so it is checked for every layout too.
                    _failures.Check($"{cell} / raw Run(CreateTensorValueFromMemory(AsDenseTensor.Memory))", () =>
                    {
                        using OrtTensor<X> h = view.AsDenseTensor<X>();
                        using OrtValue x = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, dims);
                        return RawRun(session, runOptions, x, backCode, dims, bytes);
                    });

                    _failures.Check($"{cell} / session.Run(dict, zeroCopyOutputs)", () =>
                    {
                        IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "X", view } },
                            casting: Oracle.RunCasting(code), zeroCopyOutputs: true);
                        using NDArray y = outputs["Y"];
                        return ExportContract.Imported(y, backCode, dims, bytes);
                    });
                }
            }

            /// <summary>Runs the session through ORT's own OrtValue API and checks its single output.</summary>
            /// <param name="session">The identity session.</param>
            /// <param name="runOptions">The run options.</param>
            /// <param name="x">The input tensor.</param>
            /// <param name="backCode">The dtype the output reads back as.</param>
            /// <param name="dims">The expected shape.</param>
            /// <param name="bytes">The expected bytes.</param>
            /// <returns><c>null</c> when the output holds, else the problem.</returns>
            private static string RawRun(InferenceSession session, RunOptions runOptions, OrtValue x, NPTypeCode backCode, long[] dims, byte[] bytes)
            {
                using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, new[] { "X" }, new[] { x }, new[] { "Y" });
                using NDArray y = outputs[0].ToNDArray();
                return ExportContract.Imported(y, backCode, dims, bytes);
            }

            /// <summary>Runs the session through ORT's legacy <see cref="NamedOnnxValue"/> API and checks its single output.</summary>
            /// <param name="session">The identity session.</param>
            /// <param name="x">The input.</param>
            /// <param name="backCode">The dtype the output reads back as.</param>
            /// <param name="dims">The expected shape.</param>
            /// <param name="bytes">The expected bytes.</param>
            /// <returns><c>null</c> when the output holds, else the problem.</returns>
            private static string LegacyRun(InferenceSession session, NamedOnnxValue x, NPTypeCode backCode, long[] dims, byte[] bytes)
            {
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { x });
                using NDArray y = results.First().ToNDArray();
                return ExportContract.Imported(y, backCode, dims, bytes);
            }
        }
    }
}
