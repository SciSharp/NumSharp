using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class NpRandomShuffleTest : TestClass
    {
        [TestMethod]
        public void Base1DTest()
        {
            var rnd = np.random.RandomState(42);
            var nd = np.arange(10);
            Console.WriteLine((string)nd);
            nd[8] = 5;
            rnd.shuffle(nd);
            Console.WriteLine((string)nd);
            nd.Should().BeOfValues(5, 6, 7, 3, 4, 1, 0, 5, 2, 9);
        }

        [TestMethod]
        public void Base4DTest()
        {
            var rnd = np.random.RandomState(42);
            var nd = arange(2, 2, 5, 5);
            var ogshape = nd.Shape.Clone();
            Console.WriteLine((string)nd);
            rnd.shuffle(nd);
            Console.WriteLine((string)nd);
            nd.Should().BeOfValues(11, 73, 16, 36, 49, 69, 46, 4, 8, 2, 67, 62, 26, 5,
                24, 39, 6, 84, 14, 9, 20, 65, 22, 91, 3, 23, 13, 61, 80, 40, 53, 31,
                50, 33, 44, 35, 1, 32, 74, 38, 99, 41, 78, 64, 7, 10, 96, 47, 48, 60,
                25, 59, 51, 21, 45, 77, 56, 81, 66, 82, 18, 92, 63, 54, 28, 43, 0, 17,
                68, 90, 42, 19, 27, 30, 75, 52, 98, 95, 55, 79, 12, 94, 58, 83, 34, 85,
                86, 87, 57, 71, 29, 70, 76, 89, 37, 88, 72, 97, 93, 15).And.Subject.Shape.Should().Be(ogshape);
        }
    }
}
