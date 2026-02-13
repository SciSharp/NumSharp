using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    public class StringApiTests
    {
        private static string hello = "hello";

        [Test]
        public void np_array_string()
        {
            var str = NDArray.FromString(hello);
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
        }

        [Test]
        public void FromString()
        {
            var str = NDArray.FromString(hello);
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
        }

        [Test]
        public void AsString()
        {
            var str = np.array('h', 'e', 'l', 'l', 'o');
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
            NDArray.AsString(str).Should().Be(hello);
        }

        [Test]
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

        [Test]
        public void GetString_Sliced()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5).reshape(5, 5)[":, 0"];
            // Column slice at offset 0 - NumPy-aligned IsSliced = (offset != 0)
            // Values and non-contiguous access are the important checks
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
            str.Shape.IsContiguous.Should().BeFalse("column slice has stride != 1");
            str.GetString().Should().Be("hello");
        }
        [Test]
        public void SetString_Sliced()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5).reshape(5, 5)[":, 0"];
            // Column slice at offset 0 - NumPy-aligned IsSliced = (offset != 0)
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
            str.Shape.IsContiguous.Should().BeFalse("column slice has stride != 1");
            str.GetString().Should().Be("hello");
            str.SetString("kekek");
            str.Should().BeOfValues("kekek".ToCharArray().Cast<object>().ToArray());
        }

        [Test]
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
