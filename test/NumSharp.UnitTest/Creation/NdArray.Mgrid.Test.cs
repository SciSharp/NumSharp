using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;
using System.Numerics;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayMGridTest
    {
        // These C# NDArray declarations were generated using ndarray-generatory.py,
        // which is located in the README.md of this NumSharp.UnitTest project
        // using the following Python code:
        /*
        aa, bb = np.mgrid[0:5, 0:3]
        cc, dd = np.mgrid[0:3, 0:5]
        ee, ff = np.mgrid[0:5, 0:5]
        cSharp.asCode2D("a53", aa)
        cSharp.asCode2D("b53", bb)
        cSharp.asCode2D("a35", cc)
        cSharp.asCode2D("b35", dd)
        cSharp.asCode2D("a55", ee)
        cSharp.asCode2D("b55", ff)
        */

        static NDArray a53 = new NDArray(new Int32[] {
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4
            }, new Shape(new int[] { 5, 3 }));

        static NDArray b53 = new NDArray(new Int32[] {
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
            0, 1, 2
            }, new Shape(new int[] { 5, 3 }));

        static NDArray a35 = new NDArray(new Int32[] {
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2
            }, new Shape(new int[] { 3, 5 }));

        static NDArray b35 = new NDArray(new Int32[] {
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4
            }, new Shape(new int[] { 3, 5 }));

        static NDArray a55 = new NDArray(new Int32[] {
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4
            }, new Shape(new int[] { 5, 5 }));

        static NDArray b55 = new NDArray(new Int32[] {
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4
            }, new Shape(new int[] { 5, 5 }));

        [TestMethod]
        public void BaseTest()
        {
            var V53 = np.arange(0, 5, 1).mgrid(np.arange(0, 3, 1));
            var V35 = np.arange(0, 3, 1).mgrid(np.arange(0, 5, 1));
            var V55 = np.arange(0, 5, 1).mgrid(np.arange(0, 5, 1));

            Assert.AreEqual(V53.Item1, a53);
            Assert.AreEqual(V53.Item2, b53);
            Assert.AreEqual(V35.Item1, a35);
            Assert.AreEqual(V35.Item2, b35);
            Assert.AreEqual(V55.Item1, a55);
            Assert.AreEqual(V55.Item2, b55);
        }
    }
}
