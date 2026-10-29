using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     API approval: the package's public surface, read by reflection, must equal the approved list below line for
    ///     line. A member added, removed, renamed or re-typed — a widened parameter, a lost <c>this</c>, a changed default
    ///     value such as the <c>"Features"</c> column name — fails here until the list is updated on purpose, so the
    ///     package's surface cannot change by accident. The formatter (<see cref="ApiSurface"/>) is shared with the ONNX
    ///     Runtime suite.
    /// </summary>
    [TestClass]
    public class ApiSurfaceTests : MLNetTestBase
    {
        /// <summary>The package assembly.</summary>
        private static readonly Assembly Package = typeof(NDArrayMLNetInterop).Assembly;

        /// <summary>The approved public surface, in the format <see cref="ApiSurface.Describe"/> produces.</summary>
        private static readonly string[] Approved =
        {
            "sealed class NumSharp.Interop.MLNet.NDArrayDataView : IDataView, IDisposable",
            "  Boolean CanShuffle { get; }",
            "  DataViewRowCursor GetRowCursor(IEnumerable<Column> columnsNeeded, Random rand = null)",
            "  DataViewRowCursor[] GetRowCursorSet(IEnumerable<Column> columnsNeeded, Int32 n, Random rand = null)",
            "  DataViewSchema Schema { get; }",
            "  Nullable<Int64> GetRowCount()",
            "  Void Dispose()",
            "static class NumSharp.Interop.MLNet.NDArrayMLNetInterop",
            "  static Boolean TryFromDataViewType(DataViewType type, out DType dtype)",
            "  static Boolean TryToDataViewType(DType dtype, out PrimitiveDataViewType type)",
            "  static Int32 LiveExports { get; }",
            "  static Int32 LiveImports { get; }",
            "  static NDArray AsNDArray<T>(this VBuffer<T> buffer)",
            "  static NDArray ToNDArray(this IDataView view, Column column)",
            "  static NDArray ToNDArray(this IDataView view, String columnName)",
            "  static NDArray ToNDArray<T>(this VBuffer<T> buffer)",
            "  static NDArrayDataView AsDataView(this NDArray source, String columnName = \"Features\")",
            "  static NDArrayDataView AsDataView(this NDArray source, String[] columnNames)",
            "  static NDArrayDataView ToDataView(this NDArray source, String columnName = \"Features\")",
            "  static NDArrayDataView ToDataView(this NDArray source, String[] columnNames)",
            "  static NPTypeCode FromDataViewType(DataViewType type)",
            "  static PrimitiveDataViewType ToDataViewType(DType dtype)",
            "  static Type ToDataViewClrType(DType dtype)",
            "  static VBuffer<T> ToVBuffer<T>(this NDArray source)",
            "static class NumSharp.Interop.MLNet.Postprocess",
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
