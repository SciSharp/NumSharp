using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NpFromBufferTest
    {
        [TestMethod]
        public void ToInt16()
        {
            short[] ints = { -100, 200, 300, 400, 500 };
            var bytes = new byte[ints.Length * sizeof(short)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.int16);
            Assert.AreEqual(nd.GetInt16(0), -100);
            Assert.AreEqual(nd.GetInt16(4), 500);
        }

        [TestMethod]
        public void ToInt32()
        {
            int[] ints = {-100, 200, 300, 400, 500};
            var bytes = new byte[ints.Length * sizeof(int)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.int32);
            Assert.AreEqual(nd.GetInt32(0), -100);
            Assert.AreEqual(nd.GetInt32(4), 500);
        }

        [TestMethod]
        public void ToInt64()
        {
            long[] ints = { -100, 200, 300, 400, 500 };
            var bytes = new byte[ints.Length * sizeof(long)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.int64);
            Assert.AreEqual(nd.GetInt64(0), -100);
            Assert.AreEqual(nd.GetInt64(4), 500);
        }

        [TestMethod]
        public void ToUInt16()
        {
            ushort[] ints = { 100, 200, 300, 400, 500 };
            var bytes = new byte[ints.Length * sizeof(ushort)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.uint16);
            Assert.AreEqual(nd.GetUInt16(0), (ushort)100);
            Assert.AreEqual(nd.GetUInt16(4), (ushort)500);
        }

        [TestMethod]
        public void ToUInt32()
        {
            uint[] ints = { 100, 200, 300, 400, 500 };
            var bytes = new byte[ints.Length * sizeof(uint)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.uint32);
            Assert.AreEqual(nd.GetUInt32(0), (uint)100);
            Assert.AreEqual(nd.GetUInt32(4), (uint)500);
        }

        [TestMethod]
        public void ToUInt64()
        {
            ulong[] ints = { 100, 200, 300, 400, 500 };
            var bytes = new byte[ints.Length * sizeof(ulong)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.uint64);
            Assert.AreEqual(nd.GetUInt64(0), (ulong)100);
            Assert.AreEqual(nd.GetUInt64(4), (ulong)500);
        }
        
        [TestMethod]
        public void ToSingle()
        {
            float[] floats = { 100.5F, 200.0F, 300.0F, 400.0F, 500.0F };
            var bytes = new byte[floats.Length * sizeof(float)];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.float32);
            Assert.AreEqual(nd.GetSingle(0), 100.5F);
            Assert.AreEqual(nd.GetSingle(4), 500.0F);
        }

        [TestMethod]
        public void ToDouble()
        {
            double[] floats = { 100.5, 200.0, 300.0, 400.0, 500.0 };
            var bytes = new byte[floats.Length * sizeof(double)];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.@double);
            Assert.AreEqual(nd.GetDouble(0), 100.5);
            Assert.AreEqual(nd.GetDouble(4), 500.0);
        }
       
    }
}
