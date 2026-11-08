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
        /// <summary>
        ///     A descriptor whose class has no NumSharp storage (datetime64, timedelta64, with any unit) is a dtype with no
        ///     ML.NET column type, like Complex: the <c>Try</c> map answers <c>false</c>, the CLR-type map answers
        ///     <c>null</c>, and the throwing map throws this package's own message. All three threw NumSharp's "has no
        ///     NumSharp storage yet" error, the <c>Try</c> form included (the ONNX Runtime bridge's maps already answered
        ///     this way). A non-native descriptor maps like its native twin, and a null dtype is <c>false</c> for
        ///     <c>Try</c> and an argument error for the other two.
        /// </summary>
        [TestMethod]
        public void DescriptorsWithoutStorage_AreUnmappedDtypes_NotErrors()
        {
            foreach (string spelling in new[] { "M8[ns]", "m8", "datetime64[10ns]", "timedelta64[s]", "M8" })
            {
                DType descriptor = np.dtype(spelling);
                Assert.AreEqual(NPTypeCode.Empty, descriptor.Meta.TypeCode, $"{spelling} has no storage lane");

                Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(descriptor, out PrimitiveDataViewType type), spelling);
                Assert.IsNull(type, spelling);
                Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(descriptor), spelling);
                NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => NDArrayMLNetInterop.ToDataViewType(descriptor), spelling);
                StringAssert.Contains(ex.Message, descriptor.name, spelling);
                StringAssert.Contains(ex.Message, "no arrays of this dtype yet", spelling);
            }

            DType bigEndian = np.dtype(">i4");
            Assert.IsFalse(bigEndian.isnative);
            Assert.IsTrue(NDArrayMLNetInterop.TryToDataViewType(bigEndian, out PrimitiveDataViewType int32));
            Assert.AreEqual(NumberDataViewType.Int32, int32, "a non-native descriptor maps like its native twin");
            Assert.AreEqual(NumberDataViewType.Int32, NDArrayMLNetInterop.ToDataViewType(bigEndian));
            Assert.AreEqual(typeof(int), NDArrayMLNetInterop.ToDataViewClrType(bigEndian));

            Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(null, out _), "Try never throws");
            Assert.ThrowsException<ArgumentNullException>(() => NDArrayMLNetInterop.ToDataViewType(null));
            Assert.ThrowsException<ArgumentNullException>(() => NDArrayMLNetInterop.ToDataViewClrType(null));
        }

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
