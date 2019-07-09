using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using NumSharp.NewStuff;
using OOMath;

namespace OMath.Test
{
    public partial class UnmanagedByteStorageTestsOperatorTests
    {
        [TestMethod]
        public void Subtract()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 1d);
            var ret = left - right;
            ret.All(d => d == 4).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_RightScalar()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var right = UnmanagedByteStorage<double>.Scalar(1);
            var ret = left - right;
            ret.All(d => d == 4).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_LeftScalar()
        {
            var left = UnmanagedByteStorage<double>.Scalar(1);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), 5d);
            var ret = left - right;
            ret.All(d => d == -4).Should().BeTrue();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_Rising()
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

            var ret = left - right;
            ret.Should().AllBeEquivalentTo(0);

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_RightScalar_Rising()
        {
            var left = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                left.SetIndex(i, i);
            }

            var right = UnmanagedByteStorage<double>.Scalar(1);
            var ret = left - right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_LeftScalar_Rising()
        {
            var left = UnmanagedByteStorage<double>.Scalar(1);
            var right = new UnmanagedByteStorage<double>(new Shape(5, 5), false);
            for (int i = 0; i < 25; i++)
            {
                right.SetIndex(i, i);
            }

            var ret = left - right;
            ret.Should().BeInDescendingOrder();

            for (int i = 0; i < 25; i++)
            {
                Console.WriteLine(ret.GetIndex(i));
            }
        }
    }
}
