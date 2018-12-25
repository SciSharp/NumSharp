using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayRollTest
    {
        [TestMethod]
        public void Base1DTest()
        {
            NDArray nd1 = new double[]{1,2,3,4,5,6,7,8,9,10};

            var nd2 = nd1.roll(2);

            double[] expNd1 = new double[]{1,2,3,4,5,6,7,8,9,10};
            double[] expNd2 = new double[]{9,10,1,2,3,4,5,6,7,8}; 

            double[] nd1_ = (Array) nd1 as double[];
            double[] nd2_ = (Array) nd2 as double[];


            Assert.IsTrue(Enumerable.SequenceEqual(nd1_,expNd1));
            Assert.IsTrue(Enumerable.SequenceEqual(nd2_,expNd2));

            nd1 = new double[]{1,2,3,4,5,6,7,8,9,10};

            nd2 = nd1.roll(-2);

            expNd1 = new double[]{1,2,3,4,5,6,7,8,9,10};
            expNd2 =  new double[]{3,4,5,6,7,8,9,10,1,2};

            nd1_ = (Array) nd1 as double[];
            nd2_ = (Array) nd2 as double[];

            Assert.IsTrue(Enumerable.SequenceEqual(nd1_,expNd1));
            Assert.IsTrue(Enumerable.SequenceEqual(nd2_,expNd2));

        }
        [TestMethod]
        public void Base2DTest()
        {
            var nd1 = np.arange(10).reshape(2,5);

            var nd2 = nd1.roll(2);

            Assert.IsTrue(Enumerable.SequenceEqual(nd1.shape,nd2.shape));


            var nd3 = nd1.roll(-2);

            Assert.IsTrue(Enumerable.SequenceEqual(nd1.shape,nd3.shape));


        }
    }
    
}
