using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    public class np_prod_tests
    {

        [Test]
        public void EmptyArray()
        {
            np.prod(np.array(new int[0])).Should().BeScalar(1);
        }

        [Test]
        public void Case1()
        {
            var a = np.prod(np.array(1f, 2f));
            a.GetValue(0).Should().Be(2);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [Test]
        public void Case1_double()
        {
            var a = np.prod(np.array(1d, 2d));
            a.GetValue(0).Should().Be(2d);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [Test]
        public void Case2()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2));
            a.GetValue(0).Should().Be(24f);
            a.Shape.IsScalar.Should().BeTrue();
        }

        [Test]
        public void Case3()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [Test]
        public void Case4()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [Test]
        public void Case4_dtype()
        {
            var a = np.prod(np.array(1f, 2f, 3, 4).reshape(2, 2), axis: 1, dtype: np.uint8);
            a.GetValue(0).Should().Be(2f);
            a.GetValue(1).Should().Be(12f);
            a.shape.Should().HaveCount(1).And.ContainInOrder(2);
            a.dtype.Should().Be<byte>();
        }

        /// <summary>
        /// Bug 75 fix verification: np.prod on boolean array should work.
        /// NumPy: prod([True, True, False, True]) = 0 (False acts as 0)
        /// NumPy: prod([True, True, True, True]) = 1
        /// Return type is int64 (NumPy 2.x behavior).
        /// </summary>
        [Test]
        public void BooleanArray_TreatsAsIntAndReturnsInt64()
        {
            // Array with False - product is 0
            var withFalse = np.array(new bool[] { true, true, false, true });
            var result1 = np.prod(withFalse);
            result1.GetInt64().Should().Be(0, "prod([T,T,F,T]) = 1*1*0*1 = 0");
            result1.typecode.Should().Be(NPTypeCode.Int64, "NumPy 2.x: prod(bool) returns int64");

            // Array with all True - product is 1
            var allTrue = np.array(new bool[] { true, true, true, true });
            var result2 = np.prod(allTrue);
            result2.GetInt64().Should().Be(1, "prod([T,T,T,T]) = 1*1*1*1 = 1");
            result2.typecode.Should().Be(NPTypeCode.Int64, "NumPy 2.x: prod(bool) returns int64");
        }
    }
}
