using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
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
            {
                var val = sh.MoveNext();
                Console.WriteLine((acc += val) + " | " + val);
            }

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
            var sh = new NDIterator<int>(np.arange(10), true);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNext;
            int i = 0;
            while (next() && i++ < 20)
                Console.WriteLine(acc += move());

            acc.Should().Be(Enumerable.Range(0, 10).Sum()*2);
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
        public void Case17_Reference()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator<int>(nd, false);
            int acc = 0;
            for (int i = 0; i < nd.size; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNextReference());

            acc.Should().Be(Enumerable.Range(0, 10).Sum());
        }

        [TestMethod]
        public void Case18_Reference()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator<int>(nd, false);
            int acc = 0;
            for (int i = 0; i < nd.size; i++, sh.HasNext())
                sh.MoveNextReference() = 1;

            sh.Reset();

            for (int i = 0; i < nd.size; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(nd.size);
        }

        [TestMethod]
        public void Case19_Reference_Autoreset()
        {
            Console.WriteLine();
            var nd = np.arange(10);
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;
            for (int i = 0; i < nd.size; i++, sh.HasNext())
                sh.MoveNextReference() = 1;

            for (int i = 0; i < nd.size; i++, sh.HasNext())
                Console.WriteLine(acc += sh.MoveNext());

            acc.Should().Be(nd.size);
        }

        [TestMethod]
        public void Case20_Sliced_Autoreset_Reference()
        {
            var sh = new NDIterator<int>(np.arange(15).reshape((3, 5))["0:2,:"], true);
            var acc = 0;
            var next = sh.HasNext;
            var move = sh.MoveNextReference;
            int i = 0;
            while (next() && i++ < 20)
                Console.WriteLine(acc += (move() = 1));

            acc.Should().Be(20);
        }

        [TestMethod]
        public void Case21_Sliced_Autoreset_Reference()
        {
            Console.WriteLine();
            var nd = np.arange(15).reshape((3, 5))["0:2,:"];
            var sh = new NDIterator<int>(nd, true);
            int acc = 0;

            for (int i = 0; i < nd.size * 10; i++, sh.HasNext())
                Console.WriteLine(acc += (sh.MoveNextReference() = 1));

            acc.Should().Be(10*10);
        }
    }
}
