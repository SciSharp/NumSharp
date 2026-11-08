using System;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Metamorphic relations: two routes to the same answer must agree, with no oracle needed for the answer itself.
    ///     Slicing commutes with crossing to ONNX Runtime and back; exporting a view equals exporting its contiguous copy;
    ///     transposing commutes with the dense-tensor verbs; round trips are idempotent; a chain of zero-copy verbs aliases one
    ///     buffer; and a real model gives the same output for a view as for its copy. Each relation runs over seeded random
    ///     views (<see cref="PropertyBasedTests.RandomView{T}"/>) of several dtypes.
    /// </summary>
    [TestClass]
    public class MetamorphicTests : OnnxTestBase
    {
        /// <summary>The seed the random views start from.</summary>
        private const int Seed = 4242;

        /// <summary>Random views per dtype and relation.</summary>
        private const int Cases = 60;

        /// <summary>
        ///     Slice-then-cross equals cross-then-slice: <c>ToOrtValue(v[s])</c> and <c>ToDenseTensor(v[s])</c> read back the
        ///     same bytes as <c>ToOrtValue(v)</c> read back and THEN sliced by NumSharp.
        /// </summary>
        [TestMethod]
        public void SlicingCommutesWithCrossing()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Relation(failures, (code, view, rng) =>
            {
                if (view.ndim == 0)
                    return null;
                string key = string.Join(", ", view.shape.Select(d => PropertyBasedTests.AxisSlice(rng, d)));
                using NDArray sliced = view[key];
                using OrtValue whole = view.ToOrtValue();
                using NDArray wholeBack = whole.ToNDArray();
                using NDArray thenSliced = wholeBack[key];
                byte[] want = Oracle.LogicalBytes(thenSliced);
                using OrtValue part = sliced.ToOrtValue();
                using NDArray partBack = part.ToNDArray();
                if (!Oracle.SameBytes(Oracle.LogicalBytes(partBack), want))
                    return $"ToOrtValue(v[{key}]) differs from ToOrtValue(v)[{key}]";
                using NDArray denseBack = ToDenseTensorBack(sliced);
                return Oracle.SameBytes(Oracle.LogicalBytes(denseBack), want) ? null : $"ToDenseTensor(v[{key}]) differs from ToOrtValue(v)[{key}]";
            }));
            failures.AssertNone("slice relation");
        }

        /// <summary>
        ///     Exporting a view equals exporting its contiguous copy: the zero-copy verb on <c>v.copy()</c> carries the same
        ///     bytes as the copying verb on <c>v</c>, and a real identity model returns the same output for both.
        /// </summary>
        [TestMethod]
        public void ExportingAViewEqualsExportingItsCopy()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Relation(failures, (code, view, rng) =>
            {
                using NDArray copy = view.copy();
                using OrtValue viaCopy = view.ToOrtValue();
                using OrtTensor shared = copy.AsOrtValue();
                if (!Oracle.SameBytes(Raw(viaCopy), Raw(shared.Value)))
                    return "AsOrtValue(v.copy()) differs from ToOrtValue(v)";
                using InferenceSession session = OpenSession(IdentityModel(code));
                using NDArray fromView = session.Run(view, casting: Oracle.RunCasting(code));
                using NDArray fromCopy = session.Run(copy, casting: Oracle.RunCasting(code));
                return Oracle.SameBytes(Oracle.LogicalBytes(fromView), Oracle.LogicalBytes(fromCopy)) ? null : "the model's output for v differs from its output for v.copy()";
            }));
            failures.AssertNone("copy relation");
        }

        /// <summary>
        ///     Transposing commutes with the dense-tensor verbs: <c>ToDenseTensor(v.T)[i, j, …]</c> equals
        ///     <c>ToDenseTensor(v)[…, j, i]</c>, and importing a column-major tensor equals importing the row-major tensor of the
        ///     reversed shape over the same memory, transposed.
        /// </summary>
        [TestMethod]
        public void TransposingCommutesWithTheDenseTensorVerbs()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Relation(failures, (code, view, rng) =>
            {
                if (view.ndim < 2)
                    return null;
                using NDArray transposed = view.T;
                using NDArray a = ToDenseTensorBack(transposed);
                using NDArray b = ToDenseTensorBack(view);
                using NDArray bT = b.T;
                if (!Oracle.SameBytes(Oracle.LogicalBytes(a), Oracle.LogicalBytes(bT)))
                    return "ToDenseTensor(v.T) differs from ToDenseTensor(v).T";
                return null;
            }));

            // Column-major import vs the reversed-shape row-major import of the same memory, transposed.
            var data = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
            var columnMajor = new DenseTensor<float>(data, new[] { 2, 3, 4 }, reverseStride: true);
            var rowMajorReversed = new DenseTensor<float>(data, new[] { 4, 3, 2 }, reverseStride: false);
            using NDArray fromColumnMajor = columnMajor.ToNDArray();
            using NDArray viewColumnMajor = columnMajor.AsNDArray();
            using NDArray fromRowMajor = rowMajorReversed.ToNDArray();
            using NDArray rowMajorT = fromRowMajor.T;
            failures.Check("column-major import", () => Oracle.SameBytes(Oracle.LogicalBytes(fromColumnMajor), Oracle.LogicalBytes(rowMajorT)) ? null : "ToNDArray differs");
            failures.Check("column-major zero-copy import", () => Oracle.SameBytes(Oracle.LogicalBytes(viewColumnMajor), Oracle.LogicalBytes(rowMajorT)) ? null : "AsNDArray differs");
            failures.AssertNone("transpose relation");
        }

        /// <summary>Round trips are idempotent: crossing a round trip's result again changes nothing, three times over.</summary>
        [TestMethod]
        public void RoundTripsAreIdempotent()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Relation(failures, (code, view, rng) =>
            {
                using OrtValue first = view.ToOrtValue();
                NDArray current = first.ToNDArray();
                byte[] want = Oracle.LogicalBytes(current);
                try
                {
                    for (int trip = 0; trip < 3; trip++)
                    {
                        using OrtValue again = current.ToOrtValue();
                        NDArray next = again.ToNDArray();
                        current.Dispose();
                        current = next;
                        if (!Oracle.SameBytes(Oracle.LogicalBytes(current), want))
                            return $"round trip {trip + 2} changed the elements";
                    }
                    return null;
                }
                finally
                {
                    current.Dispose();
                }
            }));
            failures.AssertNone("round-trip relation");
        }

        /// <summary>
        ///     A chain of zero-copy verbs aliases ONE buffer: NDArray → <c>AsOrtValue</c> → <c>AsNDArray</c> → <c>AsDenseTensor</c>
        ///     → <c>AsNDArray</c>, and a write through the last view is visible in the first array at the same index.
        /// </summary>
        [TestMethod]
        public void ZeroCopyChains_AliasOneBuffer()
        {
            foreach (NPTypeCode code in ZeroCopyDtypes.Where(c => c != NPTypeCode.Boolean))
            {
                using NDArray nd = Arange(code, 3, 4);
                using OrtTensor h1 = nd.AsOrtValue();
                using NDArray v1 = h1.Value.AsNDArray();
                using NDArray v2 = DenseViewOf(v1, code, out IDisposable h2);
                using (h2)
                {
                    using NDArray one = NDArray.Scalar(1).astype(code);
                    v2[2, 3] = one;
                    using NDArray written = nd["2, 3"];
                    Assert.IsTrue(Oracle.SameBytes(Oracle.LogicalBytes(written), Oracle.LogicalBytes(one)), $"{code}: the write through the chain's last view did not reach the source");
                    Assert.IsTrue(Oracle.SameBytes(Oracle.LogicalBytes(nd), Oracle.LogicalBytes(v2)), $"{code}: the chain reads a different buffer");
                }
            }
        }

        /// <summary>
        ///     <c>AsDenseTensor</c> then <c>AsNDArray</c> of <paramref name="nd"/>, dispatched on the dtype's ORT element type.
        /// </summary>
        /// <param name="nd">The array to share.</param>
        /// <param name="code">Its dtype.</param>
        /// <param name="handle">The export handle, which the caller disposes after the view.</param>
        /// <returns>The NumSharp view over the shared tensor.</returns>
        private static NDArray DenseViewOf(NDArray nd, NPTypeCode code, out IDisposable handle)
        {
            switch (code)
            {
                case NPTypeCode.Half:
                {
                    OrtTensor<Float16> h = nd.AsDenseTensor<Float16>();
                    handle = h;
                    return h.Tensor.AsNDArray();
                }
                default:
                    return DenseViewOfGeneric(nd, code, out handle);
            }
        }

        /// <summary>The generic arm of <see cref="DenseViewOf"/> for the dtypes ORT carries as their own CLR type.</summary>
        /// <param name="nd">The array to share.</param>
        /// <param name="code">Its dtype.</param>
        /// <param name="handle">The export handle.</param>
        /// <returns>The view.</returns>
        private static NDArray DenseViewOfGeneric(NDArray nd, NPTypeCode code, out IDisposable handle)
        {
            switch (code)
            {
                case NPTypeCode.Byte: { var h = nd.AsDenseTensor<byte>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.SByte: { var h = nd.AsDenseTensor<sbyte>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.Int16: { var h = nd.AsDenseTensor<short>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.UInt16: { var h = nd.AsDenseTensor<ushort>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.Int32: { var h = nd.AsDenseTensor<int>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.UInt32: { var h = nd.AsDenseTensor<uint>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.Int64: { var h = nd.AsDenseTensor<long>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.UInt64: { var h = nd.AsDenseTensor<ulong>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.Single: { var h = nd.AsDenseTensor<float>(); handle = h; return h.Tensor.AsNDArray(); }
                case NPTypeCode.Double: { var h = nd.AsDenseTensor<double>(); handle = h; return h.Tensor.AsNDArray(); }
                default: throw new ArgumentOutOfRangeException(nameof(code), code, "no ORT element type of its own");
            }
        }

        /// <summary>The raw bytes of an ORT tensor (empty for an empty tensor).</summary>
        /// <param name="v">The tensor.</param>
        /// <returns>A copy of its bytes.</returns>
        private static byte[] Raw(OrtValue v)
            => v.GetTensorTypeAndShape().ElementCount == 0 ? Array.Empty<byte>() : v.GetTensorMutableRawData().ToArray();

        /// <summary><c>ToDenseTensor</c> then <c>ToNDArray</c>, dispatched on the dtype's ORT element type (Complex: not reached).</summary>
        /// <param name="view">The view.</param>
        /// <returns>The round-tripped array.</returns>
        private static NDArray ToDenseTensorBack(NDArray view) => view.typecode switch
        {
            NPTypeCode.Boolean => view.ToDenseTensor<bool>().ToNDArray(),
            NPTypeCode.Byte => view.ToDenseTensor<byte>().ToNDArray(),
            NPTypeCode.SByte => view.ToDenseTensor<sbyte>().ToNDArray(),
            NPTypeCode.Int16 => view.ToDenseTensor<short>().ToNDArray(),
            NPTypeCode.UInt16 => view.ToDenseTensor<ushort>().ToNDArray(),
            NPTypeCode.Char => view.ToDenseTensor<ushort>().ToNDArray(),
            NPTypeCode.Int32 => view.ToDenseTensor<int>().ToNDArray(),
            NPTypeCode.UInt32 => view.ToDenseTensor<uint>().ToNDArray(),
            NPTypeCode.Int64 => view.ToDenseTensor<long>().ToNDArray(),
            NPTypeCode.UInt64 => view.ToDenseTensor<ulong>().ToNDArray(),
            NPTypeCode.Half => view.ToDenseTensor<Float16>().ToNDArray(),
            NPTypeCode.Single => view.ToDenseTensor<float>().ToNDArray(),
            NPTypeCode.Double => view.ToDenseTensor<double>().ToNDArray(),
            NPTypeCode.Decimal => view.ToDenseTensor<double>().ToNDArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(view), view.typecode, "no ONNX element type"),
        };

        /// <summary>Runs one relation over seeded random views of every dtype that crosses (Complex excluded).</summary>
        private sealed class Relation : IDtypeCheck
        {
            private readonly Failures _failures;
            private readonly Func<NPTypeCode, NDArray, Random, string> _relation;

            /// <summary>Creates the runner.</summary>
            /// <param name="failures">Where failing cells go.</param>
            /// <param name="relation">The relation: <c>null</c> when it holds for the view, else the problem.</param>
            internal Relation(Failures failures, Func<NPTypeCode, NDArray, Random, string> relation)
            {
                _failures = failures;
                _relation = relation;
            }

            /// <summary>Checks the relation over <see cref="Cases"/> random views of one dtype.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                if (code == NPTypeCode.Complex)
                    return;
                var rng = new Random(Seed ^ (int)code);
                for (int i = 0; i < Cases; i++)
                {
                    var recipe = new StringBuilder();
                    NDArray view = PropertyBasedTests.RandomView(rng, value, recipe);
                    _failures.Check($"{code} #{i} {recipe}", () => _relation(code, view, rng));
                }
            }
        }
    }
}
