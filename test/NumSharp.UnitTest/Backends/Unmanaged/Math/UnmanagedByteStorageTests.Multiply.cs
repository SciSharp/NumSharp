using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    public partial class UnmanagedByteStorageTestsOperatorTests
    {
        [TestMethod]
        public void Multiply()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 2d);
            var ret = left * right;
            ret.All(d => d == 10).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Multiply_RightScalar()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = UnmanagedByteStorage<double>.Scalar(2);
            var ret = left * right;
            ret.All(d => d == 10).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Multiply_LeftScalar()
        {
            var left = UnmanagedByteStorage<double>.Scalar(-2);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var ret = left * right;
            ret.All(d => d == -10).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Multiply_Rising()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                left.SetIndex(i, i);
            }

            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 1d);
            for (int i = 0; i < 25; i++)
            {
                right.SetIndex(i, i);
            }

            var ret = left * right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Multiply_RightScalar_Rising()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                left.SetIndex(i, i);
            }

            var right = UnmanagedByteStorage<double>.Scalar(1);
            var ret = left * right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Multiply_LeftScalar_Rising()
        {
            var left = UnmanagedByteStorage<double>.Scalar(2);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                right.SetIndex(i, i);
            }

            var ret = left * right;
            ret.Should().BeInAscendingOrder();
            ret.GetIndex(0).Should().Be(0);
            ret.GetIndex(ret.Count - 1).Should().Be(48);
            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }
    }
}
