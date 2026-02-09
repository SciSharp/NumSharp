using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    public class NdArrayScalarTests
    {
        [Test]
        //TODO! [Arguments(typeof(Complex), 3)]
        [Arguments(typeof(Boolean), (Boolean)false)]
        [Arguments(typeof(Byte), (Byte)1)]
        [Arguments(typeof(Int16), (Int16)1)]
        [Arguments(typeof(UInt16), (UInt16)1)]
        [Arguments(typeof(Int32), (Int32)1)]
        [Arguments(typeof(UInt32), (UInt32)1)]
        [Arguments(typeof(Int64), (Int64)1)]
        [Arguments(typeof(UInt64), (UInt64)1)]
        [Arguments(typeof(Char), (Char)'c')]
        [Arguments(typeof(Double), (Double)1d)]
        [Arguments(typeof(Single), (Single)2f)]
        [Arguments(typeof(Decimal), 3)]
        //TODO! [Arguments(typeof(String), "3")]
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
