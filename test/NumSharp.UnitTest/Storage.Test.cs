using NumSharp;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class StorageTester
    {
        public UnmanagedStorage strg1D;
        public UnmanagedStorage strg2D;
        public UnmanagedStorage strg2DNonFull;

        public StorageTester()
        {
            strg1D = new UnmanagedStorage(np.float64) {Engine = BackendFactory.GetEngine()};
            strg1D.Allocate(new Shape(10));
            strg1D.ReplaceData(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

            strg2D = new UnmanagedStorage(np.int64) {Engine = BackendFactory.GetEngine()};
            strg2D.Allocate(new Shape(3, 3));
            strg2D.ReplaceData(new long[] {0, 1, 2, 3, 4, 5, 6, 7, 8});

            strg2DNonFull = new UnmanagedStorage(np.float32) {Engine = BackendFactory.GetEngine()};
            strg2DNonFull.Allocate(new Shape(5, 2));
            strg2DNonFull.ReplaceData(new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
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
            Assert.IsTrue(strg1D.GetData().Count == 10);
            Assert.IsTrue(strg2D.GetData().Count == 9);
            Assert.IsTrue(strg2DNonFull.GetData().Count == 10);
        }

        [TestMethod]
        public void IndexingCheck()
        {
            var element1D = strg1D.GetValue<double>(0);
            Assert.IsTrue(element1D == 0);
            for (int idx = 1; idx < 10; idx++)
            {
                element1D = strg1D.GetValue<double>(idx);
                Assert.IsTrue(element1D == idx);
            }
        }

        [TestMethod]
        public unsafe void CloneCheck()
        {
            var l = strg1D;
            var r = strg1D.Clone();

            ReferenceEquals(l, r).Should().BeFalse();
            l.DType.Should().Be(r.DType);
            (l.InternalArray.Address != r.InternalArray.Address).Should().BeTrue();
            l.Count.Should().Be(r.Count);
            l.Shape.Should().Be(r.Shape);
        }

        [TestMethod, Ignore("Transpose is not implemented")]
        public void ReshapeLayout2d()
        {
            //var x = np.arange(6).MakeGeneric<int>();
            //var y = x.reshape((3, 2), order: 'F').MakeGeneric<int>();
            //string str = y.ToString();
            //Assert.AreEqual(y[1, 1], 4);
            //y[1, 1] = 8;
            //Assert.AreEqual(y[1, 1], 8);
            //Assert.AreEqual(x[4], 8);

            //x = np.arange(6).reshape(2, 3).MakeGeneric<int>();
            //y = x.reshape((3, 2), order: 'F').MakeGeneric<int>();
            //Assert.AreEqual(y[0, 1], 4);
            //Assert.AreEqual(y[1, 0], 3);
        }

        [TestMethod, Ignore("Transpose is not implemented")]
        public void ReshapeLayout3d()
        {
            //var x = np.arange(12).MakeGeneric<int>();
            //var y = x.reshape((2, 3, 2), order: 'F').MakeGeneric<int>();
            //string str = y.ToString();
            //Assert.AreEqual(y[1, 1], 4);
            //y[1, 1] = 8;
            //Assert.AreEqual(y[1, 1], 8);
            //Assert.AreEqual(x[4], 8);

            //x = np.arange(6).reshape(2, 3).MakeGeneric<int>();
            //y = x.reshape((3, 2), order: 'F').MakeGeneric<int>();
            //Assert.AreEqual(y[0, 1], 4);
        }

        [TestMethod]
        public void CastingViaGet()
        {
            new Action(() =>
            {
                strg1D.DType.Should().Be<Double>();
                ArraySlice<float> result = strg1D.GetData<float>();
            }).Should().NotThrow();
        }

        [Ignore]
        [TestMethod]
        public void CheckChangeTensorLayout2D()
        {
            var strg2DCpy = (UnmanagedStorage)strg2D.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions, new int[] {3, 3}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] {0, 3, 6, 1, 4, 7, 2, 5, 8}));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions, new int[] {3, 3}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), strg2D.GetData<long>()));

            strg2DCpy = (UnmanagedStorage)strg2DNonFull.Clone();

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions, new int[] {5, 2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] {0, 5, 1, 6, 2, 7, 3, 8, 4, 9}));

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.Shape.Dimensions, new int[] {5, 2}));
            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), strg2DNonFull.GetData<long>()));

            strg2DCpy = new UnmanagedStorage(typeof(long));
            strg2DCpy.Allocate(new Shape(5, 2));

            strg2DCpy.ReplaceData(strg2DNonFull.GetData());

            Assert.IsTrue(Enumerable.SequenceEqual(strg2DCpy.GetData<long>(), new long[] {0, 2, 4, 6, 8, 1, 3, 5, 7, 9}));
        }
    }
}
