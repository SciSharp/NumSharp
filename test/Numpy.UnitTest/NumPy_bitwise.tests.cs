using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Numpy.Models;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = NUnit.Framework.Assert;

namespace Numpy.UnitTest
{
    [TestClass]
    public class NumPybitwiseTest : BaseTestCase
    {
        [TestMethod]
        public void packbitsTest()
        {
            // >>> a = np.array([[[1,0,1],
            // ...                [0,1,0]],
            // ...               [[1,1,0],
            // ...                [0,0,1]]])
            // >>> b = np.packbits(a, axis=-1)
            // >>> b
            // array([[[160],[64]],[[192],[32]]], dtype=uint8)
            // 

            var a= np.array(new [,,] {{{1,0,1},
                                                { 0,1,0}},
                                                {{1,1,0},
                                                { 0,0,1}}});
            var  b = np.packbits(a, axis:-1);
            var given=  b;
            var expected=
                "array([[[160],\n        [ 64]],\n\n       [[192],\n        [ 32]]], dtype=uint8)";
            Assert.AreEqual(expected, given.repr);
            // Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,
            // and 32 = 0010 0000.            
        }

        [TestMethod]
        public void unpackbitsTest()
        {
            // >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
            // >>> a
            // array([[ 2],
            //        [ 7],
            //        [23]], dtype=uint8)
            // >>> b = np.unpackbits(a, axis=1)
            // >>> b
            // array([[0, 0, 0, 0, 0, 0, 1, 0],
            //        [0, 0, 0, 0, 0, 1, 1, 1],
            //        [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
            // 
            
            NDarray a = np.array(new[,]{{(byte)2}, { (byte)7 }, { (byte)23 }}, dtype:np.uint8);
            var given=  a;
            var expected=
                "array([[ 2],\n" +
                "       [ 7],\n" +
                "       [23]], dtype=uint8)";
            Assert.AreEqual(expected, given.repr);
            var  b = np.unpackbits(a, axis:1);
            given=  b;
            expected=
                "array([[0, 0, 0, 0, 0, 0, 1, 0],\n" +
                "       [0, 0, 0, 0, 0, 1, 1, 1],\n" +
                "       [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)";
            Assert.AreEqual(expected, given.repr);
        }

        [TestMethod]
        public void binary_reprTest()
        {
            // >>> np.binary_repr(3)
            // '11'
            // >>> np.binary_repr(-3)
            // '-11'
            // >>> np.binary_repr(3, width=4)
            // '0011'
            // 

            var given=  np.binary_repr(3);
            var expected=
                "11";
            Assert.AreEqual(expected, given);
            given=  np.binary_repr(-3);
            expected=
                "-11";
            Assert.AreEqual(expected, given);
            given=  np.binary_repr(3, width:4);
            expected=
                "0011";
            Assert.AreEqual(expected, given);

            // The two’s complement is returned when the input number is negative and
            // width is specified:
            
            // >>> np.binary_repr(-3, width=3)
            // '101'
            // >>> np.binary_repr(-3, width=5)
            // '11101'
            // 
            
            given=  np.binary_repr(-3, width:3);
            expected=
                "101";
            Assert.AreEqual(expected, given);
            given=  np.binary_repr(-3, width:5);
            expected=
                "11101";
            Assert.AreEqual(expected, given);
        }
    }
}
