using System;
using System.Linq;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Unmanaged
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
            // Column slice at offset 0 - NumPy-aligned IsSliced = (offset != 0)
            // Values and non-contiguous access are the important checks
            str.Should().BeOfType<char>().And.BeShaped(5).And.BeOfValues('h', 'e', 'l', 'l', 'o');
            str.Shape.IsContiguous.Should().BeFalse("column slice has stride != 1");
            str.GetString().Should().Be("hello");
        }
        [TestMethod]
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

        [TestMethod]
        public void SetString()
        {
            var str = np.repeat(np.array('h', 'e', 'l', 'l', 'o'), 5);
            str.Should().BeOfType<char>().And.BeShaped(5 * 5);
            str = str.reshape(5, 5);

            // SetString writes the characters of the text into a char row. (SetData(hello, 3) is NumPy's assignment of
            // the 0-d <U5 text "hello" instead: a char array is NumPy's uint16, whose setitem parses the text with int(),
            // so "hello" raises int()'s ValueError, probed against NumPy 2.4.2.)
            str.SetString(hello, 3);
            str.GetString(3).Should().Be(hello);
            new Action(() => { str.SetString(hello, 3, 1); }).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => str.SetData(hello, 3)).Should().Throw<ValueError>()
                .WithMessage("invalid literal for int() with base 10: 'hello'");
        }
    }
}
