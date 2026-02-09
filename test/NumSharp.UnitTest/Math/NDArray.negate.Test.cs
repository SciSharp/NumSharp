using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Maths
{
    public class NDArrayNegateTest : TestClass
    {
        [Test]
        public void NegateArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] {1, -2, 3.3f});

            //perform test
            nd = -nd;

            //assertions
            nd.Data<float>().Should().BeEquivalentTo(new float[] {-1, 2, -3.3f});
        }

        [Test]
        public void NegateArray2()
        {
            //initialization
            var nd = new NDArray(np.int32, 3);
            nd.ReplaceData(new int[] {-1, 0, 1});

            //perform test
            nd = -nd;

            //assertions
            nd.Data<int>().Should().BeEquivalentTo(new int[] {1, 0, -1});
        }

        [Test]
        public void NegateArray3()
        {
            //initialization
            var nd = new NDArray(np.uint32, 3);
            nd.ReplaceData(new uint[] {0, 1, 2});

            //perform test
            nd = -nd;
            Console.WriteLine(nd.ToString(false));

            //assertions
            nd.Data<uint>().Should().BeEquivalentTo(new uint[] {0, 4294967295, 4294967294});
        }

        [Test]
        public void NegateArray4()
        {
            //initialization
            var nd = new NDArray(np.uint64, 3);
            nd.ReplaceData(new ulong[] {0, 1, 2});

            //perform test
            nd = -nd;
            Console.WriteLine(nd.ToString(false));

            //assertions
            nd.Data<ulong>().Should().BeEquivalentTo(new ulong[] {0, 18446744073709551615, 18446744073709551614});
        }

        [Test]
        public void AddArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            var input = new float[] {1, -2, 3.3f};
            nd.ReplaceData(input);

            //perform test
            nd = +nd;

            //assertions
            nd.Data<float>().Should().BeEquivalentTo(new float[] {1, -2, 3.3f});
        }

        [Test]
        public void NegateEmptyArray()
        {
            //initialization
            var nd1 = new NDArray(np.float32, 0);

            //perform test
            var nd2 = -nd1;

            //assertions
            Assert.IsTrue(nd1.size == nd2.size);
        }
    }
}
