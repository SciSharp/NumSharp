using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class DtypeMapEdgeTests : MLNetTestBase
    {
        [TestMethod]
        public void ToDataViewClrType_MapsEveryDtype()
        {
            Assert.AreEqual(typeof(bool), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Boolean));
            Assert.AreEqual(typeof(byte), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Byte));
            Assert.AreEqual(typeof(sbyte), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.SByte));
            Assert.AreEqual(typeof(short), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Int16));
            Assert.AreEqual(typeof(ushort), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.UInt16));
            Assert.AreEqual(typeof(int), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Int32));
            Assert.AreEqual(typeof(uint), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.UInt32));
            Assert.AreEqual(typeof(long), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Int64));
            Assert.AreEqual(typeof(ulong), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.UInt64));
            Assert.AreEqual(typeof(ushort), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Char));   // UTF-16 code units
            Assert.AreEqual(typeof(float), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Single));
            Assert.AreEqual(typeof(double), NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Double));
            // no ML.NET column type:
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Half));
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Decimal));
            Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(NPTypeCode.Complex));
        }

        [TestMethod]
        public void TryToDataViewType_EveryDtype()
        {
            var direct = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Single, NPTypeCode.Double,
            };
            foreach (NPTypeCode c in direct)
                Assert.IsTrue(NDArrayMLNetInterop.TryToDataViewType(c, out PrimitiveDataViewType t) && t is not null, $"{c} maps");
            foreach (NPTypeCode c in new[] { NPTypeCode.Half, NPTypeCode.Decimal, NPTypeCode.Complex })
                Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(c, out _), $"{c} has no column type");
        }

        [TestMethod]
        public void TryFromDataViewType_EveryScalarColumnType()
        {
            var map = new Dictionary<DataViewType, NPTypeCode>
            {
                { BooleanDataViewType.Instance, NPTypeCode.Boolean },
                { NumberDataViewType.Byte, NPTypeCode.Byte },
                { NumberDataViewType.SByte, NPTypeCode.SByte },
                { NumberDataViewType.Int16, NPTypeCode.Int16 },
                { NumberDataViewType.UInt16, NPTypeCode.UInt16 },
                { NumberDataViewType.Int32, NPTypeCode.Int32 },
                { NumberDataViewType.UInt32, NPTypeCode.UInt32 },
                { NumberDataViewType.Int64, NPTypeCode.Int64 },
                { NumberDataViewType.UInt64, NPTypeCode.UInt64 },
                { NumberDataViewType.Single, NPTypeCode.Single },
                { NumberDataViewType.Double, NPTypeCode.Double },
            };
            foreach (KeyValuePair<DataViewType, NPTypeCode> kv in map)
                Assert.AreEqual(kv.Value, NDArrayMLNetInterop.FromDataViewType(kv.Key), $"{kv.Key}");
        }

        [TestMethod]
        public void FromDataViewType_UnwrapsVectorColumns()
        {
            Assert.AreEqual(NPTypeCode.Single, NDArrayMLNetInterop.FromDataViewType(new VectorDataViewType(NumberDataViewType.Single, 4)));
            Assert.AreEqual(NPTypeCode.Boolean, NDArrayMLNetInterop.FromDataViewType(new VectorDataViewType(BooleanDataViewType.Instance)));
            Assert.AreEqual(NPTypeCode.Int64, NDArrayMLNetInterop.FromDataViewType(new VectorDataViewType(NumberDataViewType.Int64, 2, 3)));
        }

        [TestMethod]
        public void FromDataViewType_KeyColumn_ReadsItsUnsignedStorageInteger()
        {
            Assert.AreEqual(NPTypeCode.UInt32, NDArrayMLNetInterop.FromDataViewType(new KeyDataViewType(typeof(uint), 5)));
            Assert.AreEqual(NPTypeCode.UInt64, NDArrayMLNetInterop.FromDataViewType(new KeyDataViewType(typeof(ulong), 5)));
        }

        [TestMethod]
        public void TryFromDataViewType_Text_Null_AndVectorOfText_AreRefused()
        {
            Assert.IsFalse(NDArrayMLNetInterop.TryFromDataViewType(TextDataViewType.Instance, out _));
            Assert.IsFalse(NDArrayMLNetInterop.TryFromDataViewType(null, out _));
            Assert.IsFalse(NDArrayMLNetInterop.TryFromDataViewType(new VectorDataViewType(TextDataViewType.Instance, 3), out _));

            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.FromDataViewType(TextDataViewType.Instance));
            StringAssert.Contains(ex.Message, "text");
        }

        [TestMethod]
        public void EnsureElementType_Mismatch_NamesTheExpectedType()
        {
            // ToVBuffer routes through the internal EnsureElementType; a Single array wants VBuffer<float>.
            using NDArray f = np.arange(3).astype(NPTypeCode.Single);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => f.ToVBuffer<int>());
            StringAssert.Contains(ex.Message, "Single");

            // A Char array wants VBuffer<ushort>, and says so.
            using NDArray c = np.array(new[] { 'a', 'b' });
            ArgumentException exc = Assert.ThrowsException<ArgumentException>(() => c.ToVBuffer<char>());
            StringAssert.Contains(exc.Message, "UInt16");
        }
    }
}
