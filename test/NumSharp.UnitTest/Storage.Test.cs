using NumSharp.Core;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class StorageTester
    {
        public NDStorage Storage1DInt {get;set;}
        public NDStorage Storage2DInt {get;set;}
        public NDStorage Storage1DDouble {get;set;}
        public StorageTester()
        {
            Storage1DInt = NDStorage.CreateByShapeAndType(typeof(int),new Shape(6));
            Storage1DInt.SetData(new int[] {0,1,2,3,4,5});

            Storage1DDouble = NDStorage.CreateByShapeAndType(typeof(double),new Shape(6));
            Storage1DDouble.SetData(new double[] {0.1,1.5,2.2,3.5,4.9,5.0});

            Storage2DInt = NDStorage.CreateByShapeAndType(typeof(int),new Shape(2,3));
            Storage2DInt.SetData(new int[] {0,1,2,3,4,5});
        }
        [TestMethod]
        public void Creation()
        {
            Assert.IsNotNull(Storage1DInt); 
            Assert.IsNotNull(Storage2DInt);

        }
        [TestMethod]
        public void Casting()
        { 
            int[] directCastInt = Storage1DInt.GetData<int>();

            double[] boxingCast = Storage1DInt.GetData<double>();

            double[] directCastDouble = Storage1DDouble.GetData<double>();

            int[] boxingToInt = Storage1DDouble.GetData<int>();
 
                        
        }

    }
}