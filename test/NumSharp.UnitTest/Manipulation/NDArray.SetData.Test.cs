using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArraySetData
    {

        [TestMethod]
        public void Case1_ND_Scalar()
        {
            var lhs = np.full(5, (3, 3));
            var rhs = (NDArray)1;
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);
            
            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n"+(string)lhs);

            lhs[0].Cast<int>().Should().AllBeEquivalentTo(1);
        }        
        
        [TestMethod]
        public void Case1_Scalar_Scalar()
        {
            var lhs = np.full(5, (3, 3));
            var rhs = (NDArray)1;
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0, 1] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0, 1].Cast<int>().Should().AllBeEquivalentTo(1);
        }       

        [TestMethod]
        public void Case1_ND_ND()
        {
            var lhs = np.full(5, (2, 1, 3, 3));
            var rhs = np.full(1, (1, 1, 3, 3));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9*2);
            rhs.Shape.size.Should().Be(9);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case2_ND_ScalaryND()
        {
            var lhs = np.full(5, (2, 1, 3, 3));
            var rhs = np.full(1, (1));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9*2);
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<int>().Should().AllBeEquivalentTo(1);
        }        
        
        [TestMethod]
        public void Case1_ND_Scalar_Sliced()
        {
            var lhs = np.full(5, (3, 3));
            lhs = lhs["0:2,:"];
            var rhs = (NDArray)1;
            lhs.size.Should().Be(2 * 3);
            lhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);
            
            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n"+(string)lhs);

            lhs[0].Cast<int>().Should().AllBeEquivalentTo(1);

        }        
        
        [TestMethod]
        public void Case1_Scalar_Scalar_Sliced()
        {
            var lhs = np.full(5, (3, 3));
            lhs = lhs["0:2,:"];
            var rhs = (NDArray)1;
            lhs.Shape.IsSliced.Should().BeTrue();
            lhs.size.Should().Be(2 * 3);
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0, 1] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0, 1].Cast<int>().Should().AllBeEquivalentTo(1);
        }       

        [TestMethod]
        public void Case1_ND_ND_Sliced()
        {
            var lhs = np.full(5, (2, 1, 3, 3));
            var slicedlhs = lhs;
            slicedlhs = slicedlhs[":1,:"];
            var rhs = np.full(1, (1, 1, 3, 3));
            slicedlhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.IsScalar.Should().BeFalse();
            slicedlhs.Shape.size.Should().Be(9);
            rhs.Shape.size.Should().Be(9);

            Console.WriteLine((string)lhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);
            slicedlhs[0] = rhs;
            Console.WriteLine((string)lhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);

            slicedlhs[0].Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case2_ND_ScalaryND_Sliced()
        {
            var lhs = np.full(5, (2, 1, 3, 3));
            lhs = lhs[":1,:"];
            var rhs = np.full(1, (1));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9);
            lhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<int>().Should().AllBeEquivalentTo(1);
        }

        //---------------

        [TestMethod]
        public void Case1_ND_Scalar_Cast()
        {
            var lhs = np.full(5d, (3, 3));
            var rhs = (NDArray)1;
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case1_Scalar_Scalar_Cast()
        {
            var lhs = np.full(5d, (3, 3));
            var rhs = (NDArray)1;
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);
            
            Console.WriteLine((string)lhs);
            lhs[0, 1] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0, 1].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case1_ND_ND_Cast()
        {
            var lhs = np.full(5d, (2, 1, 3, 3));
            var rhs = np.full(1, (1, 1, 3, 3));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9 * 2);
            rhs.Shape.size.Should().Be(9);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case2_ND_ScalaryND_Cast()
        {
            var lhs = np.full(5d, (2, 1, 3, 3));
            var rhs = np.full(1, (1));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9 * 2);
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case1_ND_Scalar_Sliced_Cast()
        {
            var lhs = np.full(5d, (3, 3));
            lhs = lhs["0:2,:"];
            var rhs = (NDArray)1;
            lhs.size.Should().Be(2 * 3);
            lhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<double>().Should().AllBeEquivalentTo(1);

        }

        [TestMethod]
        public void Case1_Scalar_Scalar_Sliced_Cast()
        {
            var lhs = np.full(5d, (3, 3));
            lhs = lhs["0:2,:"];
            var rhs = (NDArray)1;
            lhs.Shape.IsSliced.Should().BeTrue();
            lhs.size.Should().Be(2 * 3);
            rhs.Shape.IsScalar.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0, 1] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0, 1].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case1_ND_ND_Sliced_Cast()
        {
            var lhs = np.full(5d, (2, 1, 3, 3));
            var slicedlhs = lhs;
            slicedlhs = slicedlhs[":1,:"];
            var rhs = np.full(1d, (1, 1, 3, 3));
            slicedlhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.IsScalar.Should().BeFalse();
            slicedlhs.Shape.size.Should().Be(9);
            rhs.Shape.size.Should().Be(9);

            Console.WriteLine((string)lhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);
            slicedlhs[0] = rhs;
            Console.WriteLine((string)lhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);
            Console.WriteLine("-----");
            Console.WriteLine((string)slicedlhs);

            slicedlhs[0].Cast<double>().Should().AllBeEquivalentTo(1);
        }

        [TestMethod]
        public void Case2_ND_ScalaryND_Sliced_Cast()
        {
            var lhs = np.full(5d, (2, 1, 3, 3));
            lhs = lhs[":1,:"];
            var rhs = np.full(1, (1));
            rhs.Shape.IsScalar.Should().BeFalse();
            lhs.Shape.size.Should().Be(9);
            lhs.Shape.IsSliced.Should().BeTrue();
            rhs.Shape.size.Should().Be(1);

            Console.WriteLine((string)lhs);
            lhs[0] = rhs;
            Console.WriteLine("\n" + (string)lhs);

            lhs[0].Cast<double>().Should().AllBeEquivalentTo(1);
        }
    }
}
