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
        public NDStorage strg1D;
        public NDStorage strg2D;
        public NDStorage strg2DNonFull;
        public NDStorage strg3D;
        public NDStorage strg3DNonFull;
        public StorageTester()
        {
            strg1D = new NDStorage(np.float64);
            strg1D.Allocate(new Shape(10));
            strg1D.SetData(new double[]{0,1,2,3,4,5,6,7,8,9});

            strg2D = new NDStorage(np.int64);
            strg2D.Allocate(new Shape(3,3));
            strg2D.SetData(new Int64[]{0,1,2,3,4,5,6,7,8});

            strg2DNonFull = new NDStorage(np.float32);
            strg2DNonFull.Allocate(new Shape(5,2));
            strg2DNonFull.SetData(new float[]{0,1,2,3,4,5,6,7,8,9});

            strg3D = new NDStorage(typeof(Complex));
            strg3D.Allocate(new Shape(2,2,2));
            strg3D.SetData(new Complex[]{1,2,3,4,5,6,7,8});

            strg3DNonFull = new NDStorage(typeof(Complex));
            strg3DNonFull.Allocate(new Shape(2,3,4));
            var puffer = new Complex[24];
            for(int idx = 1;idx < 25;idx++)
                puffer[idx-1] = new Complex(idx,0);

            strg3DNonFull.SetData(puffer);

        }

        [TestMethod]
        public void Creation()
        {
            Assert.IsNotNull(strg1D); 
            Assert.IsNotNull(strg2D);
            Assert.IsNotNull(strg2DNonFull);
            Assert.IsNotNull(strg3D);
        }

        [TestMethod]
        public void InternalArrayCheck()
        {
            Assert.IsTrue(strg1D.GetData().Length == 10);
            Assert.IsTrue(strg2D.GetData().Length == 9);
            Assert.IsTrue(strg2DNonFull.GetData().Length == 10);
        }

        [Ignore]
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

            var element2D = strg2D.GetData<Int64>(0,0);
            Assert.IsTrue(element2D == 0);
            element2D = strg2D.GetData<Int64>(1,0);
            Assert.IsTrue(element2D == 1);
            element2D = strg2D.GetData<Int64>(2,0);
            Assert.IsTrue(element2D == 2);
            element2D = strg2D.GetData<Int64>(0,1);
            Assert.IsTrue(element2D == 3);
            element2D = strg2D.GetData<Int64>(1,1);
            Assert.IsTrue(element2D == 4);
            element2D = strg2D.GetData<Int64>(2,1);
            Assert.IsTrue(element2D == 5);
            element2D = strg2D.GetData<Int64>(0,2);
            Assert.IsTrue(element2D == 6);
            element2D = strg2D.GetData<Int64>(1,2);
            Assert.IsTrue(element2D == 7);
            element2D = strg2D.GetData<Int64>(2,2);
            Assert.IsTrue(element2D == 8);

            var element3d = strg3D.GetData<Complex>(0,0,0);
            element3d = strg3D.GetData<Complex>(1,0,0);
            element3d = strg3D.GetData<Complex>(0,1,0);
            element3d = strg3D.GetData<Complex>(1,1,0);

            element3d = strg3D.GetData<Complex>(0,0,1);
            element3d = strg3D.GetData<Complex>(1,0,1);
            element3d = strg3D.GetData<Complex>(0,1,1);
            element3d = strg3D.GetData<Complex>(1,1,1);
        }

        [TestMethod]
        public void CloneCheck()
        {
            var strg1DCpy = (NDStorage)strg1D.Clone();

            Assert.IsTrue(strg1DCpy.DType == strg1DCpy.GetData().GetType().GetElementType());
            Assert.IsFalse(strg1D.GetData() == strg1DCpy.GetData());
            Assert.IsTrue(strg1D.GetData().Length == strg1DCpy.GetData().Length);
            
            Assert.IsTrue(Enumerable.SequenceEqual(strg1DCpy.GetData<double>(),strg1D.GetData<double>()));
        }

        [TestMethod]
        public void ReshapeLayout()
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
            var strg2DCpy = (NDStorage)strg2D.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{3,3}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<Int64>(), new Int64[]{0,3,6,1,4,7,2,5,8} ));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{3,3}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<Int64>(), strg2D.GetData<Int64>() ));

            strg2DCpy = (NDStorage) strg2DNonFull.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{5,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<Int64>(), new Int64[]{0,5,1,6,2,7,3,8,4,9} ));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions,new int[]{5,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<Int64>(), strg2DNonFull.GetData<Int64>() ));

            strg2DCpy = new NDStorage(typeof(Int64));
            strg2DCpy.Allocate(new Shape(5,2));

            strg2DCpy.SetData(strg2DNonFull.GetData());

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<Int64>(),new Int64[]{0,2,4,6,8,1,3,5,7,9}));
        }

        [Ignore]
        [TestMethod]
        public void CheckChangeTensorLayout3D()
        {
            var strg3DCpy = (NDStorage) strg3D.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.Shape.Dimensions,new int[]{2,2,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.GetData<Complex>(), new Complex[]{1,5,3,7,2,6,4,8} ));

            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.Shape.Dimensions,new int[]{2,2,2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.GetData<Complex>(), strg3D.GetData<Complex>() ));

            strg3DCpy = (NDStorage)strg3DNonFull.Clone();

            var expectedValues = new Complex[]{1,7,13,19, 3,9,15,21, 5,11,17,23, 2,8,14,20, 4,10,16,22, 6,12,18,24 };

            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.Shape.Dimensions,new int[]{2,3,4}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.GetData<Complex>(), expectedValues ));

            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.Shape.Dimensions,new int[]{2,3,4}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg3DCpy.GetData<Complex>(), strg3DNonFull.GetData<Complex>() ));
        }
    }
}
