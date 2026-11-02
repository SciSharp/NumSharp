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

        /// <summary>
        ///     Every text array - fixed-width <c>U</c> and <c>S</c>, <c>StringDType</c>, a 2-D <c>U</c> array and an
        ///     <c>NDCharArray</c> - is refused by every export verb with the bridge's own text message, which names the
        ///     route that works. They share the <see cref="NPTypeCode.String"/> storage lane, which does not convert to a
        ///     <see cref="DType"/>: before the refusal in <c>FeedTypeCode</c>, each verb failed inside that implicit
        ///     conversion with NumSharp's "asked for a dtype by storage lane" error (for a <c>StringDType</c> array it
        ///     even called the array fixed-width), which tells the caller to pass a dtype they never asked for.
        ///     The dtype map's own refusal of a text dtype says the same.
        /// </summary>
        [TestMethod]
        public void TextArrays_AllExportVerbs_RefuseWithTheBridgesTextMessage()
        {
            using NDArray u = np.array(new[] { "alpha", "be", "" });
            using NDArray s = u.astype(np.dtype("S5"));
            using NDArray t = np.array(new[] { "alpha", "be", "" }, np.dtypes.StringDType());
            using NDArray u2 = np.array(new[,] { { "a", "b" }, { "c", "d" } });
            using NDArray chars = np.@char.array(new[] { "a", "b" });

            foreach (NDArray text in new[] { u, s, t, u2, chars })
            {
                string[] names = text.ndim == 1 ? new[] { "c0" } : new[] { "c0", "c1" };
                AssertTextRefusal(() => text.AsDataView("Text"), text, "AsDataView(column)");
                AssertTextRefusal(() => text.AsDataView(names), text, "AsDataView(columns)");
                AssertTextRefusal(() => text.ToDataView("Text"), text, "ToDataView(column)");
                AssertTextRefusal(() => text.ToDataView(names), text, "ToDataView(columns)");
                AssertTextRefusal(() => text.ToVBuffer<ushort>(), text, "ToVBuffer");
                AssertTextRefusal(() => NDArrayMLNetInterop.ToDataViewType(text.dtype), text, "ToDataViewType");
                Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(text.dtype, out _), $"TryToDataViewType({text.dtype})");
                Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(text.dtype), $"ToDataViewClrType({text.dtype})");
            }
        }

        /// <summary>Assert one export of a text array throws the bridge's text refusal, not NumSharp's storage-lane error.</summary>
        /// <param name="export">The export call.</param>
        /// <param name="text">The text array exported (named in a failure).</param>
        /// <param name="verb">The verb, named in a failure.</param>
        private static void AssertTextRefusal(Action export, NDArray text, string verb)
        {
            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(export, $"{verb} of {text.dtype}");
            StringAssert.Contains(ex.Message, "text arrays (fixed-width U / S and StringDType)", $"{verb} of {text.dtype}");
            StringAssert.Contains(ex.Message, "NDArray.AsStringArray", $"{verb} of {text.dtype}: the message names the route that works");
            Assert.IsFalse(ex.Message.Contains("storage lane"), $"{verb} of {text.dtype}: NumSharp's storage-lane error leaked out: {ex.Message}");
        }
    }
}
