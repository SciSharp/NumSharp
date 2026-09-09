using System;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class ErrorTests : MLNetTestBase
    {
        // ---- null sources ---------------------------------------------------------------------------

        [TestMethod]
        public void NullSource_AllExportVerbs_Throw()
        {
            NDArray a = null;
            Assert.ThrowsException<ArgumentNullException>(() => a.AsDataView("F"));
            Assert.ThrowsException<ArgumentNullException>(() => a.AsDataView(new[] { "x" }));
            Assert.ThrowsException<ArgumentNullException>(() => a.ToDataView("F"));
            Assert.ThrowsException<ArgumentNullException>(() => a.ToDataView(new[] { "x" }));
            Assert.ThrowsException<ArgumentNullException>(() => a.ToVBuffer<float>());
        }

        [TestMethod]
        public void NullView_ToNDArray_Throws()
        {
            IDataView v = null;
            Assert.ThrowsException<ArgumentNullException>(() => v.ToNDArray("c"));
        }

        [TestMethod]
        public void NullColumnName_ToNDArray_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.ThrowsException<ArgumentNullException>(() => dv.ToNDArray((string)null));
        }

        // ---- bad column names -----------------------------------------------------------------------

        [TestMethod]
        public void NullOrEmptyColumnName_Vector_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView((string)null));
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(""));
            Assert.ThrowsException<ArgumentException>(() => a.ToDataView(""));
        }

        [TestMethod]
        public void NullOrEmptyNameArray_ScalarColumns_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView((string[])null));
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new string[0]));
        }

        [TestMethod]
        public void NullOrEmptyNameElement_ScalarColumns_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 2);
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "a", null }));
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "a", "" }));
        }

        [TestMethod]
        public void NameCountMismatch_ScalarColumns_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 4);
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "a", "b" }));       // too few
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "a", "b", "c", "d", "e" })); // too many
        }

        [TestMethod]
        public void UnknownColumn_ToNDArray_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => dv.ToNDArray("Missing"));
            StringAssert.Contains(ex.Message, "Features");   // names the available columns
        }

        // ---- unsupported dtypes ---------------------------------------------------------------------

        [TestMethod]
        public void Complex_AllExportVerbs_Throw()
        {
            using NDArray a = np.arange(4).astype(NPTypeCode.Complex).reshape(2, 2);
            Assert.ThrowsException<NotSupportedException>(() => a.AsDataView("F"));
            Assert.ThrowsException<NotSupportedException>(() => a.AsDataView(new[] { "a", "b" }));
            Assert.ThrowsException<NotSupportedException>(() => a.ToDataView("F"));
        }
    }
}
