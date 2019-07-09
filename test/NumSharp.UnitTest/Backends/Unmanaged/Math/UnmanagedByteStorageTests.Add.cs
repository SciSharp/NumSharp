using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using OOMath;

namespace OMath.Test
{
    [TestClass]
    public partial class UnmanagedByteStorageTestsOperatorTests
    {
        [TestMethod]
        public void Add()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 1d);
            var ret = left + right;
            ret.All(d => d == 6).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Add_RightScalar()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = UnmanagedByteStorage<double>.Scalar(1);
            var ret = left + right;
            ret.All(d => d == 6).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Add_LeftScalar()
        {
            var left = UnmanagedByteStorage<double>.Scalar(1);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var ret = left + right;
            ret.All(d => d == 6).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Add_Rising()
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

            var ret = left + right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Add_RightScalar_Rising()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                left.SetIndex(i, i);
            }

            var right = UnmanagedByteStorage<double>.Scalar(1);
            var ret = left + right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Add_LeftScalar_Rising()
        {
            var left = UnmanagedByteStorage<double>.Scalar(1);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                right.SetIndex(i, i);
            }

            var ret = left + right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }
    }
}
