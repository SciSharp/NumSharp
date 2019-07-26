using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomRandintTest
    {
        [TestMethod]
        public void randint()
        {
            var a = np.random.RandomState().randint(low: 0, high: 10, size: new Shape(5, 5));
            Assert.IsTrue(a.Data<int>().Count(x => x < 10) == 25);
        }

        /// <summary>
        /// Based on issue https://github.com/SciSharp/NumSharp/issues/292
        /// </summary>
        [TestMethod]
        public void randint_2()
        {
            for (int i = 0; i < 50; i++)
            {
                var result_1 = np.random.randint(2, size: (Shape)10); // 10 numbers between [2, int.MaxValue)
                result_1.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 2).Should().BeTrue();
                result_1.Array.As<ArraySlice<int>>().Should().HaveCount(10);

                var result_2 = np.random.randint(5, size: new Shape(2, 4)); // 8 numbers between [5, int.MaxValue)
                result_2.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 5).Should().BeTrue();
                result_2.Array.As<ArraySlice<int>>().Should().HaveCount(2 * 4);

                var result_3 = np.random.randint(5, size: new Shape(2, 4)); // 2x4 matrix with elements between [5, int.MaxValue)
                result_3.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 5).Should().BeTrue();
                result_3.Array.As<ArraySlice<int>>().Should().HaveCount(2 * 4);

                var result_4 = np.random.randint(low: 0, high: 5); // throws System.NullReferenceException
                result_4.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 5).Should().BeTrue();
                result_4.Array.As<ArraySlice<int>>().Should().HaveCount(1);

                var result_5 = np.random.randint(0, 5, null); // throws System.NullReferenceException (equivalent to result_4)
                result_5.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 5).Should().BeTrue();
                result_5.Array.As<ArraySlice<int>>().Should().HaveCount(1);

                var result_6 = np.random.randint(5); // Does not even compile
                result_6.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 5).Should().BeTrue();
                result_6.Array.As<ArraySlice<int>>().Should().HaveCount(1);

                var result_7 = np.random.randint(low: 0, high: 10, size: new Shape(2, 2)); // 2x2 matrix with elements between [0, 10)
                result_7.Array.As<ArraySlice<int>>().All(v => v >= 0 && v < 10).Should().BeTrue();
                result_7.Array.As<ArraySlice<int>>().Should().HaveCount(2 * 2);

                var result_8 = np.random.randint(1, 5, (Shape) 8); // 8 numbers between [1, 5)
                result_8.Array.As<ArraySlice<int>>().All(v => v >= 1 && v < 5).Should().BeTrue();
                result_8.Array.As<ArraySlice<int>>().Should().HaveCount(8);

                var result_9 = np.random.randint(1, 5, new Shape(3, 2)); // 3x2 matrix with elements between [1, 5)
                result_9.Array.As<ArraySlice<int>>().All(v => v >= 1 && v < 5).Should().BeTrue();
                result_9.Array.As<ArraySlice<int>>().Should().HaveCount(3 * 2);
            }
        }
    }
}
