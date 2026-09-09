using System;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class DtypeMapTests : MLNetTestBase
    {
        [TestMethod]
        public void ToDataViewType_MapsTheElevenDirectDtypes()
        {
            Assert.AreEqual(BooleanDataViewType.Instance, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Boolean));
            Assert.AreEqual(NumberDataViewType.Byte, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Byte));
            Assert.AreEqual(NumberDataViewType.SByte, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.SByte));
            Assert.AreEqual(NumberDataViewType.Int16, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Int16));
            Assert.AreEqual(NumberDataViewType.UInt16, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.UInt16));
            Assert.AreEqual(NumberDataViewType.Int32, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Int32));
            Assert.AreEqual(NumberDataViewType.UInt32, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.UInt32));
            Assert.AreEqual(NumberDataViewType.Int64, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Int64));
            Assert.AreEqual(NumberDataViewType.UInt64, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.UInt64));
            Assert.AreEqual(NumberDataViewType.Single, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Single));
            Assert.AreEqual(NumberDataViewType.Double, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Double));
        }

        [TestMethod]
        public void Char_MapsToUInt16_OneWay()
        {
            Assert.AreEqual(NumberDataViewType.UInt16, NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Char));
            Assert.AreEqual(typeof(ushort), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Char));
            // Directional: a UInt16 column reads back as UInt16, never Char.
            Assert.AreEqual(NPTypeCode.UInt16, NDArrayMLNetInterop.FromDataViewType(NumberDataViewType.UInt16));
        }

        [TestMethod]
        public void FromDataViewType_MapsScalarAndVectorColumns()
        {
            Assert.AreEqual(NPTypeCode.Single, NDArrayMLNetInterop.FromDataViewType(NumberDataViewType.Single));
            Assert.AreEqual(NPTypeCode.Boolean, NDArrayMLNetInterop.FromDataViewType(BooleanDataViewType.Instance));
            // A vector column unwraps to its item type.
            var vec = new VectorDataViewType(NumberDataViewType.Double, 4);
            Assert.AreEqual(NPTypeCode.Double, NDArrayMLNetInterop.FromDataViewType(vec));
        }

        [TestMethod]
        public void HalfDecimalComplex_HaveNoColumnType_LowLevelMapRefuses()
        {
            Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(NPTypeCode.Half, out _));
            Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(NPTypeCode.Decimal, out _));
            Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(NPTypeCode.Complex, out _));
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Half));
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Decimal));
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Complex));

            NotSupportedException half = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Half));
            StringAssert.Contains(half.Message, "Single");
            NotSupportedException dec = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Decimal));
            StringAssert.Contains(dec.Message, "Double");
            NotSupportedException cplx = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.ToDataViewType(NPTypeCode.Complex));
            StringAssert.Contains(cplx.Message, "complex");
        }

        [TestMethod]
        public void FromDataViewType_RefusesText()
        {
            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.FromDataViewType(TextDataViewType.Instance));
            StringAssert.Contains(ex.Message, "text");
            Assert.IsFalse(NDArrayMLNetInterop.TryFromDataViewType(TextDataViewType.Instance, out _));
            Assert.IsFalse(NDArrayMLNetInterop.TryFromDataViewType(null, out _));
        }

        [TestMethod]
        public void RoundTrip_EveryDirectDtype()
        {
            foreach (NPTypeCode code in DirectDtypes)
            {
                PrimitiveDataViewType t = NDArrayMLNetInterop.ToDataViewType(code);
                Assert.AreEqual(code, NDArrayMLNetInterop.FromDataViewType(t), $"round-trip {code}");
            }
        }
    }
}
