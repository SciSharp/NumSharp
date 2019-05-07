using NumSharp;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class StorageTester
    {
        public ArrayStorage strg1D;
        public ArrayStorage strg2D;
        public ArrayStorage strg2DNonFull;
        public StorageTester()
        {
            strg1D = new ArrayStorage(np.float64);
            strg1D.Allocate(new Shape(10));
            strg1D.SetData(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            strg2D = new ArrayStorage(np.int64);
            strg2D.Allocate(new Shape(3,3));
            strg2D.SetData(new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });

            strg2DNonFull = new ArrayStorage(np.float32);
            strg2DNonFull.Allocate(new Shape(5,2));
            strg2DNonFull.SetData(new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        }

        [TestMethod]
        public void Creation()
        {
            Assert.IsNotNull(strg1D); 
            Assert.IsNotNull(strg2D);
            Assert.IsNotNull(strg2DNonFull);
        }

        [TestMethod]
        public void InternalArrayCheck()
        {
            Assert.IsTrue(strg1D.GetData().Length == 10);
            Assert.IsTrue(strg2D.GetData().Length == 9);
            Assert.IsTrue(strg2DNonFull.GetData().Length == 10);
        }

        [TestMethod]
        public void IndexingCheck()
        {
            var element1D = strg1D.GetData<double>(0);
            Assert.IsTrue(element1D == 0);
            for (int idx = 1; idx < 10;idx++)
            {
                element1D = strg1D.GetData<double>(idx);
                Assert.IsTrue(element1D == idx);
            }
        }

        [TestMethod]
        public void CloneCheck()
        {
            var strg1DCpy = (ArrayStorage)strg1D.Clone();

            Assert.IsTrue(strg1DCpy.DType == strg1DCpy.GetData().GetType().GetElementType());
            Assert.IsFalse(strg1D.GetData() == strg1DCpy.GetData());
            Assert.IsTrue(strg1D.GetData().Length == strg1DCpy.GetData().Length);
            
            Assert.IsTrue(Enumerable.SequenceEqual(strg1DCpy.GetData<double>(),strg1D.GetData<double>()));
        }

        [TestMethod]
        public void ReshapeLayout2d()
        {
            var x = np.arange(6).MakeGeneric<int>();
            var y = x.reshape((3, 2), order: "F").MakeGeneric<int>();
            string str = y.ToString();
            Assert.AreEqual(y[1, 1], 4);
            y[1, 1] = 8;
            Assert.AreEqual(y[1, 1], 8);
            Assert.AreEqual(x[4], 8);

            x = np.arange(6).reshape(2, 3).MakeGeneric<int>();
            y = x.reshape((3, 2), order: "F").MakeGeneric<int>();
            Assert.AreEqual(y[0, 1], 4);
            Assert.AreEqual(y[1, 0], 3);
        }

        [Ignore]
        [TestMethod]
        public void ReshapeLayout3d()
        {
            var x = np.arange(12).MakeGeneric<int>();
            var y = x.reshape((2, 3, 2), order: "F").MakeGeneric<int>();
            string str = y.ToString();
            Assert.AreEqual(y[1, 1], 4);
            y[1, 1] = 8;
            Assert.AreEqual(y[1, 1], 8);
            Assert.AreEqual(x[4], 8);

            x = np.arange(6).reshape(2, 3).MakeGeneric<int>();
            y = x.reshape((3, 2), order: "F").MakeGeneric<int>();
            Assert.AreEqual(y[0, 1], 4);
        }

        [TestMethod]
        public void CastingViaGet()
        {
            double[] arr1 = strg1D.GetData<double>();
        }

        [Ignore]
        [TestMethod]
        public void CheckChangeTensorLayout2D()
        {
            var strg2DCpy = (ArrayStorage)strg2D.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions, new int[] { 3, 3 }));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] { 0, 3, 6, 1, 4, 7, 2, 5, 8 }));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{3,3}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), strg2D.GetData<long>() ));

            strg2DCpy = (ArrayStorage) strg2DNonFull.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{5,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] { 0, 5, 1, 6, 2, 7, 3, 8, 4, 9 }));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{5,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), strg2DNonFull.GetData<long>()));

            strg2DCpy = new ArrayStorage(typeof(long));
            strg2DCpy.Allocate(new Shape(5,2));

            strg2DCpy.SetData(strg2DNonFull.GetData());

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 }));
        }
    }
}
