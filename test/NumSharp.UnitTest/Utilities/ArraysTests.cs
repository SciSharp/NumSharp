using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Utilities
{
    public class ArraysTests
    {
        [Test]
        public void Create_1()
        {
            Arrays.Create(typeof(int), 1000).Should().BeOfType<int[]>().Which.Should().HaveCount(1000);
        }

        [Test]
        public void Create_2()
        {
            Arrays.Create(typeof(int), new int[] {1000}).Should().BeOfType<int[]>().Which.Should().HaveCount(1000);
        }

        [Test]
        public void Create_3()
        {
            Arrays.Create(NPTypeCode.Int32, 1000).Should().BeOfType<int[]>().Which.Should().HaveCount(1000);
        }

        [Test]
        public void Insert_0()
        {
            int index = 0;
            var a = Enumerable.Range(0, 10).ToArray();
            Arrays.Insert(ref a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_2()
        {
            int index = 2;
            var a = Enumerable.Range(0, 10).ToArray();
            Arrays.Insert(ref a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_5()
        {
            int index = 5;
            var a = Enumerable.Range(0, 10).ToArray();
            Arrays.Insert(ref a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_9()
        {
            int index = 9;
            var a = Enumerable.Range(0, 10).ToArray();
            Arrays.Insert(ref a, index, 3);
            a[index].Should().Be(3);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_0_Copy()
        {
            int index = 0;
            var a = Enumerable.Range(0, 10).ToArray();
            a = Arrays.Insert(a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_2_Copy()
        {
            int index = 2;
            var a = Enumerable.Range(0, 10).ToArray();
            a = Arrays.Insert(a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_5_Copy()
        {
            int index = 5;
            var a = Enumerable.Range(0, 10).ToArray();
            a = Arrays.Insert(a, index, 9);
            a[index].Should().Be(9);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }

        [Test]
        public void Insert_9_Copy()
        {
            int index = 9;
            var a = Enumerable.Range(0, 10).ToArray();
            a = Arrays.Insert(a, index, 3);
            a[index].Should().Be(3);
            var l = a.ToList();
            l.RemoveAt(index);
            Enumerable.SequenceEqual(l, Enumerable.Range(0, 10)).Should().BeTrue();
        }
    }
}
