using System;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayScalarTests
    {
        [DataTestMethod]
        //TODO! [DataRow(typeof(Complex), 3)]
        [DataRow(typeof(Boolean), (Boolean)false)]
        [DataRow(typeof(Byte), (Byte)1)]
        [DataRow(typeof(Int16), (Int16)1)]
        [DataRow(typeof(UInt16), (UInt16)1)]
        [DataRow(typeof(Int32), (Int32)1)]
        [DataRow(typeof(UInt32), (UInt32)1)]
        [DataRow(typeof(Int64), (Int64)1)]
        [DataRow(typeof(UInt64), (UInt64)1)]
        [DataRow(typeof(Char), (Char)'c')]
        [DataRow(typeof(Double), (Double)1d)]
        [DataRow(typeof(Single), (Single)2f)]
        [DataRow(typeof(Decimal), 3)]
        //TODO! [DataRow(typeof(String), "3")]
        public void CreateScalar(Type type, object val)
        {
            if (type == typeof(Complex))
                val = new Complex(1d, 2d);
            if (type == typeof(Decimal))
                val = 3m;

            var sc = NDArray.Scalar((ValueType)val);
            sc.ndim.Should().Be(0);
            sc.size.Should().Be(1);
            sc.Array.GetIndex(0).Should().Be(val);
        }
    }
}
