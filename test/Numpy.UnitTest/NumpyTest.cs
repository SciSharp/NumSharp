using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Numpy;
using Numpy.Models;
using Assert = NUnit.Framework.Assert;

namespace Numpy.UnitTests
{
    [TestClass]
    public class NumpyTest
    {
        [TestMethod]
        public void empty()
        {
            // initialize an array with random integers
            var a = np.empty(new Shape(2, 3), np.int32);
            Console.WriteLine(a.repr);
            Assert.IsNotNull(a.ToString());
            // this should print out the exact integers of the array
            foreach (var x in a.GetData<int>())
                Console.WriteLine(x);
        }

        [TestMethod]
        public void efficient_array_copy()
        {
            var a = np.empty(new Shape(2, 3), np.int32);
            Console.WriteLine(a.repr);
            Assert.IsNotNull(a.ToString());
            long ptr = a.PyObject.ctypes.data;
            Console.WriteLine("ptr: " + ptr);
            var array = new int[] { 1, 2, 3, 4, 5, 6 };
            Marshal.Copy(array, 0, new IntPtr(ptr), array.Length);
            Console.WriteLine(a.ToString());
        }

        [TestMethod]
        public void array()
        {
            var array = new int[] { 1, 2, 3, 4, 5, 6 };
            var a = np.array(array);
            Console.WriteLine(a.repr);
            Assert.AreEqual(array, a.GetData());
        }

        [TestMethod]
        public void ndarray_shape()
        {
            var array = new int[] { 1, 2, 3, 4, 5, 6 };
            var a = np.array(array);
            Assert.AreEqual(new Shape(6), a.shape);
            Assert.AreEqual(new Shape(100), np.arange(100).shape);
        }

        [TestMethod]
        public void ndarray_strides()
        {
            Assert.AreEqual(new int[] { 4 }, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).strides);
            Assert.AreEqual(new int[] { 8 }, np.arange(10, dtype: np.longlong).strides);
        }

        [TestMethod]
        public void ndarray_ndim()
        {
            Assert.AreEqual(1, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).ndim);
            Assert.AreEqual(1, np.arange(10, dtype: np.longlong).ndim);
        }

        [TestMethod]
        public void ndarray_size()
        {
            Assert.AreEqual(6, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).size);
            Assert.AreEqual(10, np.arange(10, dtype: np.longlong).size);
        }

        [TestMethod]
        public void ndarray_len()
        {
            Assert.AreEqual(6, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).len);
            Assert.AreEqual(10, np.arange(10, dtype: np.longlong).len);
        }

        [TestMethod]
        public void ndarray_itemsize()
        {
            Assert.AreEqual(4, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).itemsize);
            Assert.AreEqual(8, np.arange(10, dtype: np.longlong).itemsize);
        }

        [TestMethod]
        public void ndarray_nbytes()
        {
            Assert.AreEqual(24, np.array(new int[] { 1, 2, 3, 4, 5, 6 }).nbytes);
            Assert.AreEqual(80, np.arange(10, dtype: np.longlong).nbytes);
        }

        [TestMethod]
        public void ndarray_base()
        {
            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
            var b = a.reshape(new Shape(2, 3));
            Assert.AreEqual(null, a.@base);
            Assert.AreEqual(a, b.@base);
        }

        [TestMethod]
        public void ndarray_dtype()
        {
            Assert.AreEqual(np.int32, np.array(new int[] { 1, 2, 3, 4, 5, 6 }, dtype: np.int32).dtype);
            Assert.AreEqual(np.longlong, np.arange(10, dtype: np.longlong).dtype);
            Assert.AreEqual(np.float32, np.arange(10, dtype: np.float32).dtype);
            Assert.AreEqual(np.@double, np.arange(10, dtype: np.float64).dtype);
        }

        [TestMethod]
        public void ndarray_multidim_source_array()
        {
            var a = np.array(new float[,] { { 1f, 2f }, { 3f, 4f }, { 3f, 4f } });
            Console.WriteLine(a.repr);
            Assert.AreEqual(new Shape(3, 2), a.shape);
            Assert.AreEqual(np.float32, a.dtype);
        }

        [TestMethod]
        public void ndarray_T()
        {
            var x = np.array(new float[,] { { 1f, 2f }, { 3f, 4f } });
            Assert.AreEqual("[[1. 2.]\n [3. 4.]]", x.ToString());
            var t = x.T;
            Console.WriteLine(t.repr);
            Assert.AreEqual("[[1. 3.]\n [2. 4.]]", t.ToString());
            Assert.AreEqual(new[] { 1f, 2f, 3f, 4f }, t.GetData<float>());
        }

        [TestMethod]
        public void ndarray_flatten()
        {
            var x = np.array(new float[,] { { 1f, 2f }, { 3f, 4f } });
            Assert.AreEqual("[1. 2. 3. 4.]", x.flatten().ToString());
            var t = x.T;
            Assert.AreEqual("[1. 3. 2. 4.]", t.flatten().ToString());
            Assert.AreEqual(new[] { 1f, 3f, 2f, 4f }, t.flatten().GetData<float>());
        }

        [TestMethod]
        public void ndarray_reshape()
        {
            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
            var b = a.reshape(new Shape(2, 3));
            Assert.AreEqual("[[1 2 3]\n [4 5 6]]", b.str);
            Assert.AreEqual(new Shape(2, 3), b.shape);
            Assert.AreEqual(null, a.@base);
            Assert.AreEqual(a, b.@base);
        }

        [TestMethod]
        public void ndarray_indexing()
        {
            var x = np.arange(10);
            Assert.AreEqual("2", x["2"].str);
            Assert.AreEqual("8", x["-2"].str);
            Assert.AreEqual("[2 3 4 5 6 7]", x["2:-2"].str);
            var y = x.reshape(new Shape(2, 5));
            Assert.AreEqual("8", y["1,3"].str);
            Assert.AreEqual("9", y["1,-1"].str);
            Assert.AreEqual("array([0, 1, 2, 3, 4])", y["0"].repr);
            Assert.AreEqual("2", y["0"]["2"].str);
        }

        [TestMethod]
        public void ndarray_indexing1()
        {
            var x = np.arange(10);
            Assert.AreEqual("2", x[2].str);
            Assert.AreEqual("8", x[-2].str);
            var y = x.reshape(new Shape(2, 5));
            Assert.AreEqual("8", y[1, 3].str);
            Assert.AreEqual("9", y[1, -1].str);
            Assert.AreEqual("array([0, 1, 2, 3, 4])", y[0].repr);
            Assert.AreEqual("2", y[0][2].str);
        }

        [TestMethod]
        public void ndarray_indexing2()
        {
            var x = np.arange(10, 1, -1);
            Assert.AreEqual("array([10,  9,  8,  7,  6,  5,  4,  3,  2])", x.repr);
            Assert.AreEqual("array([7, 7, 9, 2])", x[np.array(new[] { 3, 3, 1, 8 })].repr);
            Assert.AreEqual("array([7, 7, 4, 2])", x[np.array(new[] { 3, 3, -3, 8 })].repr);
            Assert.AreEqual("array([[9, 9],\n       [8, 7]])", x[np.array(new int[,] { { 1, 1 }, { 2, 3 } })].repr);
        }

        [TestMethod]
        public void ndarray_indexing3()
        {
            var y = np.arange(35).reshape(5, 7);
            Assert.AreEqual("array([ 0, 15, 30])", y[np.array(0, 2, 4), np.array(0, 1, 2)].repr);
            Assert.AreEqual("array([ 1, 15, 29])", y[np.array(0, 2, 4), 1].repr);
            Assert.AreEqual(
                "array([[ 0,  1,  2,  3,  4,  5,  6],\n       [14, 15, 16, 17, 18, 19, 20],\n       [28, 29, 30, 31, 32, 33, 34]])",
                y[np.array(0, 2, 4)].repr);
        }

        [TestMethod]
        public void ndarray_slice()
        {
            var x = np.arange(10);
            Assert.AreEqual("array([2, 3, 4])", x["2:5"].repr);
            Assert.AreEqual("array([0, 1, 2])", x[":-7"].repr);
            Assert.AreEqual("array([1, 3, 5])", x["1:7:2"].repr);
            var y = np.arange(35).reshape(new Shape(5, 7));
            Assert.AreEqual("array([[ 7, 10, 13],\n       [21, 24, 27]])", y["1:5:2,::3"].repr);
        }

        [TestMethod]
        public void ndarray_slice1()
        {
            var y = np.arange(35).reshape(5, 7);
            var b = y > 20;
            Assert.AreEqual(
                "array([[ 1,  2],\n" +
                "       [15, 16],\n" +
                "       [29, 30]])",
                y[np.array(0, 2, 4), "1:3"].repr);
            Assert.AreEqual("array([[22, 23],\n       [29, 30]])", y[b[":", 5], "1:3"].repr);
        }

        [TestMethod]
        public void ndarray_masking()
        {
            var y = np.arange(35).reshape(5, 7);
            var b = y > 20;
            Assert.AreEqual("array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])", y[b].repr);
            // use a 1-D boolean whose first dim agrees with the first dim of y
            Assert.AreEqual("array([False, False, False,  True,  True])", b[":", 5].repr);
            Assert.AreEqual("array([[21, 22, 23, 24, 25, 26, 27],\n       [28, 29, 30, 31, 32, 33, 34]])", y[b[":", 5]].repr);
        }

        [TestMethod]
        public void ndarray_masking1()
        {
            var x = np.arange(30).reshape(2, 3, 5);
            Assert.AreEqual(
                "array([[[ 0,  1,  2,  3,  4],\n" +
                "        [ 5,  6,  7,  8,  9],\n" +
                "        [10, 11, 12, 13, 14]],\n\n" +
                "       [[15, 16, 17, 18, 19],\n" +
                "        [20, 21, 22, 23, 24],\n" +
                "        [25, 26, 27, 28, 29]]])",
                 x.repr);
            var b = np.array(new[,] {{true, true, false}, {false, true, true}});
            Assert.AreEqual(
                "array([[ 0,  1,  2,  3,  4],\n"+
                "       [ 5,  6,  7,  8,  9],\n" +
                "       [20, 21, 22, 23, 24],\n" +
                "       [25, 26, 27, 28, 29]])",
            x[b].repr);
        }

        // TODO:  https://docs.scipy.org/doc/numpy/user/basics.indexing.html?highlight=slice#structural-indexing-tools
        // TODO:  https://docs.scipy.org/doc/numpy/user/basics.indexing.html?highlight=slice#assigning-values-to-indexed-arrays
        // TODO:  https://docs.scipy.org/doc/numpy/user/basics.indexing.html?highlight=slice#dealing-with-variable-numbers-of-indices-within-programs
    }
}
