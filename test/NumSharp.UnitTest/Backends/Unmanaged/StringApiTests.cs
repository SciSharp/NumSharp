using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class StringApiTests
    {
        private static string hello = "hello";

        [TestMethod]
        public void np_array_string()
        {
            var str = NDArray.FromString(hello);
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
        }

        [TestMethod]
        public void FromString()
        {
            var str = NDArray.FromString(hello);
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
        }

        [TestMethod]
        public void AsString()
        {
            var str = np.array('h', 'e', 'l', 'l', 'o');
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
            NDArray.AsString(str).Should().Be(hello);
        }

        [TestMethod]
        public void GetString()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5);
            str.Should().BeOfType<char>().And.BeShaped(5 * 5);
            str = str.reshape(5, 5);
            str.GetString(3).Should().Be("lllll");
            new Action(() =>
            {
                var _ = str.GetString(3, 1);
            }).Should().Throw<ArgumentOutOfRangeException>();
        }

        [TestMethod]
        public void GetString_Sliced()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5).reshape(5, 5)[":, 0"];
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o').And.BeSliced();
            str.GetString().Should().Be("hello");
        }
        [TestMethod]
        public void SetString_Sliced()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5).reshape(5, 5)[":, 0"];
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o').And.BeSliced();
            str.GetString().Should().Be("hello");
            str.SetString("kekek");
            str.Should().BeOfValues("kekek".ToCharArray().Cast<object>().ToArray());
        }

        [TestMethod]
        public void SetString()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5);
            str.Should().BeOfType<char>().And.BeShaped(5 * 5);
            str = str.reshape(5, 5);

            str.SetData(hello, 3);
            str.GetString(3).Should().Be(hello);
            new Action(() => { str.SetString(hello, 3, 1); }).Should().Throw<ArgumentOutOfRangeException>();
        }
    }
}
