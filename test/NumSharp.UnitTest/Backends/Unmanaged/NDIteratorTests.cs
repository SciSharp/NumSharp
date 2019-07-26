using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class NDIteratorTests
    {
        [TestMethod]
        public void Case1()
        {
            var sh = new NDIterator<int>(np.arange(10), false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case2()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;
            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case3_Sliced()
        {
            var sh = new NDIterator<int>(np.arange(15).reshape((3, 5))["0:2,:"], false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case4_Sliced()
        {
            Console.WriteLine();
            var nd = np.arange(15).reshape((3, 5))["0:2,:"];
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;

            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case5_Autoreset()
        {
            var sh = new NDIterator<int>(np.arange(10), false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case6_Autoreset()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;
            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case7_Sliced_Autoreset()
        {
            var sh = new NDIterator<int>(np.arange(15).reshape((3, 5))["0:2,:"], false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case8_Sliced_Autoreset()
        {
            Console.WriteLine();
            var nd = np.arange(15).reshape((3, 5))["0:2,:"];
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;

            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case9_Nongeneric()
        {
            var sh = new NDIterator(np.arange(10), false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += (int)move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case10_Nongeneric()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator(nd, true);
            int acc = 0;
            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += (int)sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case11_Sliced_Nongeneric()
        {
            var sh = new NDIterator(np.arange(15).reshape((3, 5))["0:2,:"], false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += (int)move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case12_Sliced_Nongeneric()
        {
            Console.WriteLine();
            var nd = np.arange(15).reshape((3, 5))["0:2,:"];
            var sh = new NDIterator(nd, true);
            int acc = 0;

            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += (int)sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case13_Autoreset_Nongeneric()
        {
            var sh = new NDIterator(np.arange(10), false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += (int)move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case14_Autoreset_Nongeneric()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator(nd, true);
            int acc = 0;
            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += (int)sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }

        [TestMethod]
        public void Case15_Sliced_Autoreset_Nongeneric()
        {
            var sh = new NDIterator(np.arange(15).reshape((3, 5))["0:2,:"], false);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            while (next())
                Console.WriteLine(acc += (int)move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case16_Sliced_Autoreset_Nongeneric()
        {
            Console.WriteLine();
            var nd = np.arange(15).reshape((3, 5))["0:2,:"];
            var sh = new NDIterator(nd, true);
            int acc = 0;

            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += (int)sh.MoveNext());

            acc.Should().Be(Enumerable.Range(0, 10).Sum() * 10);
        }
    }
}
