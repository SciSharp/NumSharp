using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     API approval: the package's public surface, read by reflection, must equal the approved list below line for
    ///     line. A member added, removed, renamed or re-typed — a widened parameter, a lost <c>this</c>, a changed default
    ///     value — fails here until the list is updated on purpose, so the package's surface cannot change by accident.
    /// </summary>
    [TestClass]
    public class ApiSurfaceTests : OnnxTestBase
    {
        /// <summary>The package assembly.</summary>
        private static readonly Assembly Package = typeof(NDArrayOnnxInterop).Assembly;

        /// <summary>The approved public surface, in the format <see cref="ApiSurface.Describe"/> produces.</summary>
        private static readonly string[] Approved =
        {
            "static class NumSharp.Interop.OnnxRuntime.InferenceSessionExtensions",
            "  static IReadOnlyDictionary<String, NDArray> Run(this InferenceSession session, IReadOnlyDictionary<String, NDArray> inputs, IReadOnlyCollection<String> outputNames = null, RunOptions runOptions = null, String casting = \"safe\", Boolean zeroCopyOutputs = false)",
            "  static NDArray Run(this InferenceSession session, NDArray input, RunOptions runOptions = null, String casting = \"safe\")",
            "  static NDArray Run(this InferenceSession session, String inputName, NDArray input, String outputName, RunOptions runOptions = null, String casting = \"safe\")",
            "  static Void Run(this InferenceSession session, IReadOnlyDictionary<String, NDArray> inputs, IReadOnlyDictionary<String, NDArray> outputs, RunOptions runOptions = null, String casting = \"safe\")",
            "static class NumSharp.Interop.OnnxRuntime.NDArrayOnnxInterop",
            "  static Boolean TryFromTensorElementType(TensorElementType type, out DType dtype)",
            "  static Boolean TryToTensorElementType(DType dtype, out TensorElementType type)",
            "  static DenseTensor<T> ToDenseTensor<T>(this NDArray source)",
            "  static Int32 LiveExports { get; }",
            "  static Int32 LiveImports { get; }",
            "  static NDArray AsNDArray(this OrtValue value, Boolean ownsValue = false, OnnxCodecMode mode = Auto)",
            "  static NDArray AsNDArray<T>(this DenseTensor<T> tensor, OnnxCodecMode mode = Auto)",
            "  static NDArray ToNDArray(this NamedOnnxValue value)",
            "  static NDArray ToNDArray(this OrtValue value)",
            "  static NDArray ToNDArray<T>(this DenseTensor<T> tensor)",
            "  static NDArray[] ToNDArrays(this OrtValue sequence)",
            "  static NPTypeCode FromTensorElementType(TensorElementType type)",
            "  static OrtTensor AsOrtValue(this NDArray source, OnnxCodecMode mode = Auto)",
            "  static OrtTensor<T> AsDenseTensor<T>(this NDArray source, OnnxCodecMode mode = Auto)",
            "  static OrtValue GetMapKeys(this OrtValue map)",
            "  static OrtValue ToOrtValue(this NDArray source)",
            "  static String[] ReadStringTensor(this OrtValue tensor)",
            "  static TensorElementType ToTensorElementType(DType dtype)",
            "  static Type ToTensorElementClrType(DType dtype)",
            "  static ValueTuple<NDArray, NDArray> ToMap(this OrtValue map)",
            "  static ValueTuple<NDArray, NDArray>[] ToMaps(this OrtValue sequenceOfMaps)",
            "enum NumSharp.Interop.OnnxRuntime.OnnxCodecMode",
            "  const OnnxCodecMode Auto = 0",
            "  const OnnxCodecMode Copy = 2",
            "  const OnnxCodecMode View = 1",
            "sealed class NumSharp.Interop.OnnxRuntime.OrtTensor : IDisposable",
            "  Boolean IsDisposed { get; }",
            "  Boolean IsView { get; }",
            "  Int64[] Shape { get; }",
            "  NDArray Source { get; }",
            "  OrtValue Value { get; }",
            "  TensorElementType ElementType { get; }",
            "  Void Dispose()",
            "sealed class NumSharp.Interop.OnnxRuntime.OrtTensor<T> : IDisposable",
            "  Boolean IsDisposed { get; }",
            "  Boolean IsView { get; }",
            "  DenseTensor<T> Tensor { get; }",
            "  Memory<T> Memory { get; }",
            "  NDArray Source { get; }",
            "  Void Dispose()",
            "static class NumSharp.Interop.OnnxRuntime.Postprocess",
            "  static NDArray Argmax(NDArray scores, Int32 axis = -1, Boolean keepdims = false)",
            "  static NDArray LogSoftmax(NDArray logits, Int32 axis = -1)",
            "  static NDArray Sigmoid(NDArray x)",
            "  static NDArray Softmax(NDArray logits, Int32 axis = -1)",
            "  static ValueTuple<NDArray, NDArray> TopK(NDArray scores, Int32 k, Int32 axis = -1, Boolean largest = true, Boolean sorted = true)",
        };

        /// <summary>The public surface, as reflection sees it, equals the approved list exactly.</summary>
        [TestMethod]
        public void PublicSurface_IsExactlyTheApprovedOne()
        {
            string[] actual = ApiSurface.Describe(Package);
            actual.Should().Equal(Approved,
                "the public API changed; if that was intended, update Approved. Actual surface:\n" + string.Join("\n", actual));
        }
    }
}
