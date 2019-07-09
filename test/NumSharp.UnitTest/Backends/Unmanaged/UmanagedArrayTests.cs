using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using OOMath;

namespace OMath.Test
{
    [TestClass]
    public class UmanagedArrayTests
    {
        [TestMethod]
        public void Slice()
        {
            var mem = new UnmanagedArray<int>(1000);
            try
            {
                for (int i = 0; i < 1000; i++)
                {
                    mem[i] = i;
                }

                var arr = new ArraySlice<int>(mem);
                var r = arr.Slice(50, 10);
                var rr = r.Slice(0, 5);

                r.ToArray().Should().ContainInOrder(Enumerable.Range(50, 10));
                rr.ToArray().Should().ContainInOrder(Enumerable.Range(50, 5));
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void Fill()
        {
            var mem = new UnmanagedArray<int>(1000);
            try
            {
                for (int i = 0; i < 1000; i++)
                {
                    mem[i] = i;
                }

                var arr = new ArraySlice<int>(mem);
                var r = arr.Slice(50, 10);
                var rr = r.Slice(0, 5);

                r.ToArray().Should().ContainInOrder(Enumerable.Range(50, 10));
                rr.ToArray().Should().ContainInOrder(Enumerable.Range(50, 5));
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void AllocateExpand()
        {
            var mem = new UnmanagedArray<int>(10);

            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i] = i;
                }

                mem.Reallocate(15, true);
                mem.Count.Should().Be(15);
                mem.ToArray().Should().ContainInOrder(Enumerable.Range(0, 10));

                //assure we can access new 5 items
                mem[13].Should().BeOfType(typeof(int));
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void AllocateExpandFill()
        {
            var mem = new UnmanagedArray<int>(10);
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i] = i;
                }

                mem.Reallocate(15, true, -5);
                mem.Count.Should().Be(15);
                mem.ToArray().Should().ContainInOrder(Enumerable.Range(0, 10).Concat(Enumerable.Repeat(-5, 5)));

                //assure we can access new 5 items
                mem[13].Should().Be(-5);
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void AllocateShrinkFill()
        {
            var mem = new UnmanagedArray<int>(10);
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i] = i;
                }

                mem.Reallocate(5, true, -5);
                mem.Count.Should().Be(5);
                mem.ToArray().Should().ContainInOrder(Enumerable.Range(0, 5));
                mem[3].Should().Be(3);
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void AllocateShrink()
        {
            var mem = new UnmanagedArray<int>(10);
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i] = i;
                }

                mem.Reallocate(5, true);
                mem.Count.Should().Be(5);
                mem.ToArray().Should().ContainInOrder(Enumerable.Range(0, 5));
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void FromManaged()
        {
            var arr = Enumerable.Range(0, 10).ToArray();
            var mem = UnmanagedArray<int>.FromArray(arr);
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i].Should().Be(i);
                }

                arr[1] = 5;
                mem[1].Should().Be(5);
            }
            finally
            {
                mem.Free();
            }
        }

        [TestMethod]
        public void FromManaged_Copy()
        {
            var arr = Enumerable.Range(0, 10).ToArray();
            var mem = UnmanagedArray<int>.FromArray(arr, copy: true);
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    mem[i].Should().Be(i);
                }

                arr[1] = 5;
                mem[1].Should().NotBe(5);
            }
            finally
            {
                mem.Free();
            }
        }


        [TestMethod]
        public void Cast()
        {
            var arr = UnmanagedArray<int>.FromArray(Enumerable.Range(0, 10).ToArray());

            UnmanagedArray<byte> cast = new UnmanagedArray<byte>();
            try
            {
                cast = UnmanagedArray.Cast<int, byte>(arr);
                cast.Should().AllBeOfType<byte>().And.BeInAscendingOrder();
            }
            finally
            {
                arr.Free();
                cast.Free();
            }
        }
    }
}
