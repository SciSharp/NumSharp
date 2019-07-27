using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayEyeTest
    {
        [TestMethod]
        public void Case1()
        {
            np.eye(3, k: 1).Cast<double>().Count(i => i == 1d).Should().Be(2);

            np.eye(3, k: 1).Cast<double>().Should()
                .BeEquivalentTo(new NDArray(new double[][] {new double[] {0.0d, 1.0d, 0.0d}, new double[] {0.0d, 0d, 1.0d}, new double[] {0d, 0d, 0d}}, Shape.Matrix(3, 3)));
        }        
        
        [TestMethod]
        public void Case2()
        {
            np.eye(3).Cast<double>().Count(i => i == 1).Should().Be(3);
            np.eye(3).Cast<double>().Should()
                .BeEquivalentTo(new NDArray(new double[][] { new double[] { 1.0d, 0.0d, 0.0d }, new double[] { 0.0d, 1d, 0d }, new double[] { 0d, 0d, 1d } }, Shape.Matrix(3, 3)));

        }        
        [TestMethod]
        public void Case3()
        {
            np.eye(10, k: -6).Cast<double>().Count(i=>i==1).Should().Be(4);
        }        
        [TestMethod]
        public void Case4()
        {
            np.eye(10, k: -6).Cast<double>().Count(i=>i==1).Should().Be(4);
        }


    }
}
