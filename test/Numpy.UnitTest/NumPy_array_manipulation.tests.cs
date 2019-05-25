using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = NUnit.Framework.Assert;

namespace Numpy.UnitTest
{
    [TestClass]
    public class NumPyarray_manipulationTest : BaseTestCase
    {
        [TestMethod]
        public void reshapeTest()
        {
            // >>> a = np.array([[1,2,3], [4,5,6]])
            // >>> np.reshape(a, 6)
            // array([1, 2, 3, 4, 5, 6])
            // >>> np.reshape(a, 6, order='F')
            // array([1, 4, 2, 5, 3, 6])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1,2,3], [4,5,6]]);
            given=  np.reshape(a, 6);
            expected=
                "array([1, 2, 3, 4, 5, 6])";
            Assert.AreEqual(expected, given.repr);
            given=  np.reshape(a, 6, order='F');
            expected=
                "array([1, 4, 2, 5, 3, 6])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
            // array([[1, 2],
            //        [3, 4],
            //        [5, 6]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2;
            expected=
                "array([[1, 2],\n" +
                "       [3, 4],\n" +
                "       [5, 6]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void ravelTest()
        {
            // It is equivalent to reshape(-1, order=order).
            
            // >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            // >>> print(np.ravel(x))
            // [1 2 3 4 5 6]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([[1, 2, 3], [4, 5, 6]]);
            given=  print(np.ravel(x));
            expected=
                "[1 2 3 4 5 6]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> print(x.reshape(-1))
            // [1 2 3 4 5 6]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  print(x.reshape(-1));
            expected=
                "[1 2 3 4 5 6]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> print(np.ravel(x, order='F'))
            // [1 4 2 5 3 6]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  print(np.ravel(x, order='F'));
            expected=
                "[1 4 2 5 3 6]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // When order is ‘A’, it will preserve the array’s ‘C’ or ‘F’ ordering:
            
            // >>> print(np.ravel(x.T))
            // [1 4 2 5 3 6]
            // >>> print(np.ravel(x.T, order='A'))
            // [1 2 3 4 5 6]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  print(np.ravel(x.T));
            expected=
                "[1 4 2 5 3 6]";
            Assert.AreEqual(expected, given.repr);
            given=  print(np.ravel(x.T, order='A'));
            expected=
                "[1 2 3 4 5 6]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // When order is ‘K’, it will preserve orderings that are neither ‘C’
            // nor ‘F’, but won’t reverse axes:
            
            // >>> a = np.arange(3)[::-1]; a
            // array([2, 1, 0])
            // >>> a.ravel(order='C')
            // array([2, 1, 0])
            // >>> a.ravel(order='K')
            // array([2, 1, 0])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.arange(3)[::-1]; a;
            expected=
                "array([2, 1, 0])";
            Assert.AreEqual(expected, given.repr);
            given=  a.ravel(order='C');
            expected=
                "array([2, 1, 0])";
            Assert.AreEqual(expected, given.repr);
            given=  a.ravel(order='K');
            expected=
                "array([2, 1, 0])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
            // array([[[ 0,  2,  4],
            //         [ 1,  3,  5]],
            //        [[ 6,  8, 10],
            //         [ 7,  9, 11]]])
            // >>> a.ravel(order='C')
            // array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
            // >>> a.ravel(order='K')
            // array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a;
            expected=
                "array([[[ 0,  2,  4],\n" +
                "        [ 1,  3,  5]],\n" +
                "       [[ 6,  8, 10],\n" +
                "        [ 7,  9, 11]]])";
            Assert.AreEqual(expected, given.repr);
            given=  a.ravel(order='C');
            expected=
                "array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])";
            Assert.AreEqual(expected, given.repr);
            given=  a.ravel(order='K');
            expected=
                "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void moveaxisTest()
        {
            // >>> x = np.zeros((3, 4, 5))
            // >>> np.moveaxis(x, 0, -1).shape
            // (4, 5, 3)
            // >>> np.moveaxis(x, -1, 0).shape
            // (5, 3, 4)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.zeros((3, 4, 5));
            given=  np.moveaxis(x, 0, -1).shape;
            expected=
                "(4, 5, 3)";
            Assert.AreEqual(expected, given.repr);
            given=  np.moveaxis(x, -1, 0).shape;
            expected=
                "(5, 3, 4)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // These all achieve the same result:
            
            // >>> np.transpose(x).shape
            // (5, 4, 3)
            // >>> np.swapaxes(x, 0, -1).shape
            // (5, 4, 3)
            // >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
            // (5, 4, 3)
            // >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
            // (5, 4, 3)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.transpose(x).shape;
            expected=
                "(5, 4, 3)";
            Assert.AreEqual(expected, given.repr);
            given=  np.swapaxes(x, 0, -1).shape;
            expected=
                "(5, 4, 3)";
            Assert.AreEqual(expected, given.repr);
            given=  np.moveaxis(x, [0, 1], [-1, -2]).shape;
            expected=
                "(5, 4, 3)";
            Assert.AreEqual(expected, given.repr);
            given=  np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape;
            expected=
                "(5, 4, 3)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void rollaxisTest()
        {
            // >>> a = np.ones((3,4,5,6))
            // >>> np.rollaxis(a, 3, 1).shape
            // (3, 6, 4, 5)
            // >>> np.rollaxis(a, 2).shape
            // (5, 3, 4, 6)
            // >>> np.rollaxis(a, 1, 4).shape
            // (3, 5, 6, 4)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.ones((3,4,5,6));
            given=  np.rollaxis(a, 3, 1).shape;
            expected=
                "(3, 6, 4, 5)";
            Assert.AreEqual(expected, given.repr);
            given=  np.rollaxis(a, 2).shape;
            expected=
                "(5, 3, 4, 6)";
            Assert.AreEqual(expected, given.repr);
            given=  np.rollaxis(a, 1, 4).shape;
            expected=
                "(3, 5, 6, 4)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void swapaxesTest()
        {
            // >>> x = np.array([[1,2,3]])
            // >>> np.swapaxes(x,0,1)
            // array([[1],
            //        [2],
            //        [3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([[1,2,3]]);
            given=  np.swapaxes(x,0,1);
            expected=
                "array([[1],\n" +
                "       [2],\n" +
                "       [3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
            // >>> x
            // array([[[0, 1],
            //         [2, 3]],
            //        [[4, 5],
            //         [6, 7]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]]);
            given=  x;
            expected=
                "array([[[0, 1],\n" +
                "        [2, 3]],\n" +
                "       [[4, 5],\n" +
                "        [6, 7]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.swapaxes(x,0,2)
            // array([[[0, 4],
            //         [2, 6]],
            //        [[1, 5],
            //         [3, 7]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.swapaxes(x,0,2);
            expected=
                "array([[[0, 4],\n" +
                "        [2, 6]],\n" +
                "       [[1, 5],\n" +
                "        [3, 7]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void transposeTest()
        {
            // >>> x = np.arange(4).reshape((2,2))
            // >>> x
            // array([[0, 1],
            //        [2, 3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(4).reshape((2,2));
            given=  x;
            expected=
                "array([[0, 1],\n" +
                "       [2, 3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.transpose(x)
            // array([[0, 2],
            //        [1, 3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.transpose(x);
            expected=
                "array([[0, 2],\n" +
                "       [1, 3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.ones((1, 2, 3))
            // >>> np.transpose(x, (1, 0, 2)).shape
            // (2, 1, 3)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.ones((1, 2, 3));
            given=  np.transpose(x, (1, 0, 2)).shape;
            expected=
                "(2, 1, 3)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void atleast_1dTest()
        {
            // >>> np.atleast_1d(1.0)
            // array([ 1.])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.atleast_1d(1.0);
            expected=
                "array([ 1.])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(9.0).reshape(3,3)
            // >>> np.atleast_1d(x)
            // array([[ 0.,  1.,  2.],
            //        [ 3.,  4.,  5.],
            //        [ 6.,  7.,  8.]])
            // >>> np.atleast_1d(x) is x
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(9.0).reshape(3,3);
            given=  np.atleast_1d(x);
            expected=
                "array([[ 0.,  1.,  2.],\n" +
                "       [ 3.,  4.,  5.],\n" +
                "       [ 6.,  7.,  8.]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.atleast_1d(x) is x;
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.atleast_1d(1, [3, 4])
            // [array([1]), array([3, 4])]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.atleast_1d(1, [3, 4]);
            expected=
                "[array([1]), array([3, 4])]";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void atleast_2dTest()
        {
            // >>> np.atleast_2d(3.0)
            // array([[ 3.]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.atleast_2d(3.0);
            expected=
                "array([[ 3.]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(3.0)
            // >>> np.atleast_2d(x)
            // array([[ 0.,  1.,  2.]])
            // >>> np.atleast_2d(x).base is x
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(3.0);
            given=  np.atleast_2d(x);
            expected=
                "array([[ 0.,  1.,  2.]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.atleast_2d(x).base is x;
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.atleast_2d(1, [1, 2], [[1, 2]])
            // [array([[1]]), array([[1, 2]]), array([[1, 2]])]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.atleast_2d(1, [1, 2], [[1, 2]]);
            expected=
                "[array([[1]]), array([[1, 2]]), array([[1, 2]])]";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void atleast_3dTest()
        {
            // >>> np.atleast_3d(3.0)
            // array([[[ 3.]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.atleast_3d(3.0);
            expected=
                "array([[[ 3.]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(3.0)
            // >>> np.atleast_3d(x).shape
            // (1, 3, 1)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(3.0);
            given=  np.atleast_3d(x).shape;
            expected=
                "(1, 3, 1)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(12.0).reshape(4,3)
            // >>> np.atleast_3d(x).shape
            // (4, 3, 1)
            // >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(12.0).reshape(4,3);
            given=  np.atleast_3d(x).shape;
            expected=
                "(4, 3, 1)";
            Assert.AreEqual(expected, given.repr);
            given=  np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself;
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
            // ...     print(arr, arr.shape)
            // ...
            // [[[1]
            //   [2]]] (1, 2, 1)
            // [[[1]
            //   [2]]] (1, 2, 1)
            // [[[1 2]]] (1, 1, 2)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):;
            expected=
                "...     print(arr, arr.shape)\n" +
                "...\n" +
                "[[[1]\n" +
                "  [2]]] (1, 2, 1)\n" +
                "[[[1]\n" +
                "  [2]]] (1, 2, 1)\n" +
                "[[[1 2]]] (1, 1, 2)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void broadcast_toTest()
        {
            // >>> x = np.array([1, 2, 3])
            // >>> np.broadcast_to(x, (3, 3))
            // array([[1, 2, 3],
            //        [1, 2, 3],
            //        [1, 2, 3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([1, 2, 3]);
            given=  np.broadcast_to(x, (3, 3));
            expected=
                "array([[1, 2, 3],\n" +
                "       [1, 2, 3],\n" +
                "       [1, 2, 3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void broadcast_arraysTest()
        {
            // >>> x = np.array([[1,2,3]])
            // >>> y = np.array([[4],[5]])
            // >>> np.broadcast_arrays(x, y)
            // [array([[1, 2, 3],
            //        [1, 2, 3]]), array([[4, 4, 4],
            //        [5, 5, 5]])]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([[1,2,3]]);
            given=  y = np.array([[4],[5]]);
            given=  np.broadcast_arrays(x, y);
            expected=
                "[array([[1, 2, 3],\n" +
                "       [1, 2, 3]]), array([[4, 4, 4],\n" +
                "       [5, 5, 5]])]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Here is a useful idiom for getting contiguous copies instead of
            // non-contiguous views.
            
            // >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
            // [array([[1, 2, 3],
            //        [1, 2, 3]]), array([[4, 4, 4],
            //        [5, 5, 5]])]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  [np.array(a) for a in np.broadcast_arrays(x, y)];
            expected=
                "[array([[1, 2, 3],\n" +
                "       [1, 2, 3]]), array([[4, 4, 4],\n" +
                "       [5, 5, 5]])]";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void expand_dimsTest()
        {
            // >>> x = np.array([1,2])
            // >>> x.shape
            // (2,)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([1,2]);
            given=  x.shape;
            expected=
                "(2,)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // The following is equivalent to x[np.newaxis,:] or x[np.newaxis]:
            
            // >>> y = np.expand_dims(x, axis=0)
            // >>> y
            // array([[1, 2]])
            // >>> y.shape
            // (1, 2)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  y = np.expand_dims(x, axis=0);
            given=  y;
            expected=
                "array([[1, 2]])";
            Assert.AreEqual(expected, given.repr);
            given=  y.shape;
            expected=
                "(1, 2)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> y = np.expand_dims(x, axis=1)  # Equivalent to x[:,np.newaxis]
            // >>> y
            // array([[1],
            //        [2]])
            // >>> y.shape
            // (2, 1)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  y = np.expand_dims(x, axis=1)  # Equivalent to x[:,np.newaxis];
            given=  y;
            expected=
                "array([[1],\n" +
                "       [2]])";
            Assert.AreEqual(expected, given.repr);
            given=  y.shape;
            expected=
                "(2, 1)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Note that some examples may use None instead of np.newaxis.  These
            // are the same objects:
            
            // >>> np.newaxis is None
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.newaxis is None;
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void squeezeTest()
        {
            // >>> x = np.array([[[0], [1], [2]]])
            // >>> x.shape
            // (1, 3, 1)
            // >>> np.squeeze(x).shape
            // (3,)
            // >>> np.squeeze(x, axis=0).shape
            // (3, 1)
            // >>> np.squeeze(x, axis=1).shape
            // Traceback (most recent call last):
            // ...
            // ValueError: cannot select an axis to squeeze out which has size not equal to one
            // >>> np.squeeze(x, axis=2).shape
            // (1, 3)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.array([[[0], [1], [2]]]);
            given=  x.shape;
            expected=
                "(1, 3, 1)";
            Assert.AreEqual(expected, given.repr);
            given=  np.squeeze(x).shape;
            expected=
                "(3,)";
            Assert.AreEqual(expected, given.repr);
            given=  np.squeeze(x, axis=0).shape;
            expected=
                "(3, 1)";
            Assert.AreEqual(expected, given.repr);
            given=  np.squeeze(x, axis=1).shape;
            expected=
                "Traceback (most recent call last):\n" +
                "...\n" +
                "ValueError: cannot select an axis to squeeze out which has size not equal to one";
            Assert.AreEqual(expected, given.repr);
            given=  np.squeeze(x, axis=2).shape;
            expected=
                "(1, 3)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void asfarrayTest()
        {
            // >>> np.asfarray([2, 3])
            // array([ 2.,  3.])
            // >>> np.asfarray([2, 3], dtype='float')
            // array([ 2.,  3.])
            // >>> np.asfarray([2, 3], dtype='int8')
            // array([ 2.,  3.])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.asfarray([2, 3]);
            expected=
                "array([ 2.,  3.])";
            Assert.AreEqual(expected, given.repr);
            given=  np.asfarray([2, 3], dtype='float');
            expected=
                "array([ 2.,  3.])";
            Assert.AreEqual(expected, given.repr);
            given=  np.asfarray([2, 3], dtype='int8');
            expected=
                "array([ 2.,  3.])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void asfortranarrayTest()
        {
            // >>> x = np.arange(6).reshape(2,3)
            // >>> y = np.asfortranarray(x)
            // >>> x.flags['F_CONTIGUOUS']
            // False
            // >>> y.flags['F_CONTIGUOUS']
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(6).reshape(2,3);
            given=  y = np.asfortranarray(x);
            given=  x.flags['F_CONTIGUOUS'];
            expected=
                "False";
            Assert.AreEqual(expected, given.repr);
            given=  y.flags['F_CONTIGUOUS'];
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Note: This function returns an array with at least one-dimension (1-d) 
            // so it will not preserve 0-d arrays.
            
        }
        [TestMethod]
        public void asarray_chkfiniteTest()
        {
            // Convert a list into an array.  If all elements are finite
            // asarray_chkfinite is identical to asarray.
            
            // >>> a = [1, 2]
            // >>> np.asarray_chkfinite(a, dtype=float)
            // array([1., 2.])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = [1, 2];
            given=  np.asarray_chkfinite(a, dtype=float);
            expected=
                "array([1., 2.])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Raises ValueError if array_like contains Nans or Infs.
            
            // >>> a = [1, 2, np.inf]
            // >>> try:
            // ...     np.asarray_chkfinite(a)
            // ... except ValueError:
            // ...     print('ValueError')
            // ...
            // ValueError
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = [1, 2, np.inf];
            given=  try:;
            expected=
                "...     np.asarray_chkfinite(a)\n" +
                "... except ValueError:\n" +
                "...     print('ValueError')\n" +
                "...\n" +
                "ValueError";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void asscalarTest()
        {
            // >>> np.asscalar(np.array([24]))
            // 24
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.asscalar(np.array([24]));
            expected=
                "24";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void requireTest()
        {
            // >>> x = np.arange(6).reshape(2,3)
            // >>> x.flags
            //   C_CONTIGUOUS : True
            //   F_CONTIGUOUS : False
            //   OWNDATA : False
            //   WRITEABLE : True
            //   ALIGNED : True
            //   WRITEBACKIFCOPY : False
            //   UPDATEIFCOPY : False
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(6).reshape(2,3);
            given=  x.flags;
            expected=
                "  C_CONTIGUOUS : True\n" +
                "  F_CONTIGUOUS : False\n" +
                "  OWNDATA : False\n" +
                "  WRITEABLE : True\n" +
                "  ALIGNED : True\n" +
                "  WRITEBACKIFCOPY : False\n" +
                "  UPDATEIFCOPY : False";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
            // >>> y.flags
            //   C_CONTIGUOUS : False
            //   F_CONTIGUOUS : True
            //   OWNDATA : True
            //   WRITEABLE : True
            //   ALIGNED : True
            //   WRITEBACKIFCOPY : False
            //   UPDATEIFCOPY : False
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F']);
            given=  y.flags;
            expected=
                "  C_CONTIGUOUS : False\n" +
                "  F_CONTIGUOUS : True\n" +
                "  OWNDATA : True\n" +
                "  WRITEABLE : True\n" +
                "  ALIGNED : True\n" +
                "  WRITEBACKIFCOPY : False\n" +
                "  UPDATEIFCOPY : False";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void concatenateTest()
        {
            // >>> a = np.array([[1, 2], [3, 4]])
            // >>> b = np.array([[5, 6]])
            // >>> np.concatenate((a, b), axis=0)
            // array([[1, 2],
            //        [3, 4],
            //        [5, 6]])
            // >>> np.concatenate((a, b.T), axis=1)
            // array([[1, 2, 5],
            //        [3, 4, 6]])
            // >>> np.concatenate((a, b), axis=None)
            // array([1, 2, 3, 4, 5, 6])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1, 2], [3, 4]]);
            given=  b = np.array([[5, 6]]);
            given=  np.concatenate((a, b), axis=0);
            expected=
                "array([[1, 2],\n" +
                "       [3, 4],\n" +
                "       [5, 6]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.concatenate((a, b.T), axis=1);
            expected=
                "array([[1, 2, 5],\n" +
                "       [3, 4, 6]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.concatenate((a, b), axis=None);
            expected=
                "array([1, 2, 3, 4, 5, 6])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // This function will not preserve masking of MaskedArray inputs.
            
            // >>> a = np.ma.arange(3)
            // >>> a[1] = np.ma.masked
            // >>> b = np.arange(2, 5)
            // >>> a
            // masked_array(data=[0, --, 2],
            //              mask=[False,  True, False],
            //        fill_value=999999)
            // >>> b
            // array([2, 3, 4])
            // >>> np.concatenate([a, b])
            // masked_array(data=[0, 1, 2, 2, 3, 4],
            //              mask=False,
            //        fill_value=999999)
            // >>> np.ma.concatenate([a, b])
            // masked_array(data=[0, --, 2, 2, 3, 4],
            //              mask=[False,  True, False, False, False, False],
            //        fill_value=999999)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.ma.arange(3);
            given=  a[1] = np.ma.masked;
            given=  b = np.arange(2, 5);
            given=  a;
            expected=
                "masked_array(data=[0, --, 2],\n" +
                "             mask=[False,  True, False],\n" +
                "       fill_value=999999)";
            Assert.AreEqual(expected, given.repr);
            given=  b;
            expected=
                "array([2, 3, 4])";
            Assert.AreEqual(expected, given.repr);
            given=  np.concatenate([a, b]);
            expected=
                "masked_array(data=[0, 1, 2, 2, 3, 4],\n" +
                "             mask=False,\n" +
                "       fill_value=999999)";
            Assert.AreEqual(expected, given.repr);
            given=  np.ma.concatenate([a, b]);
            expected=
                "masked_array(data=[0, --, 2, 2, 3, 4],\n" +
                "             mask=[False,  True, False, False, False, False],\n" +
                "       fill_value=999999)";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void stackTest()
        {
            // >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
            // >>> np.stack(arrays, axis=0).shape
            // (10, 3, 4)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  arrays = [np.random.randn(3, 4) for _ in range(10)];
            given=  np.stack(arrays, axis=0).shape;
            expected=
                "(10, 3, 4)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.stack(arrays, axis=1).shape
            // (3, 10, 4)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.stack(arrays, axis=1).shape;
            expected=
                "(3, 10, 4)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.stack(arrays, axis=2).shape
            // (3, 4, 10)
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.stack(arrays, axis=2).shape;
            expected=
                "(3, 4, 10)";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> a = np.array([1, 2, 3])
            // >>> b = np.array([2, 3, 4])
            // >>> np.stack((a, b))
            // array([[1, 2, 3],
            //        [2, 3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([1, 2, 3]);
            given=  b = np.array([2, 3, 4]);
            given=  np.stack((a, b));
            expected=
                "array([[1, 2, 3],\n" +
                "       [2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.stack((a, b), axis=-1)
            // array([[1, 2],
            //        [2, 3],
            //        [3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.stack((a, b), axis=-1);
            expected=
                "array([[1, 2],\n" +
                "       [2, 3],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void column_stackTest()
        {
            // >>> a = np.array((1,2,3))
            // >>> b = np.array((2,3,4))
            // >>> np.column_stack((a,b))
            // array([[1, 2],
            //        [2, 3],
            //        [3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array((1,2,3));
            given=  b = np.array((2,3,4));
            given=  np.column_stack((a,b));
            expected=
                "array([[1, 2],\n" +
                "       [2, 3],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void dstackTest()
        {
            // >>> a = np.array((1,2,3))
            // >>> b = np.array((2,3,4))
            // >>> np.dstack((a,b))
            // array([[[1, 2],
            //         [2, 3],
            //         [3, 4]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array((1,2,3));
            given=  b = np.array((2,3,4));
            given=  np.dstack((a,b));
            expected=
                "array([[[1, 2],\n" +
                "        [2, 3],\n" +
                "        [3, 4]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> a = np.array([[1],[2],[3]])
            // >>> b = np.array([[2],[3],[4]])
            // >>> np.dstack((a,b))
            // array([[[1, 2]],
            //        [[2, 3]],
            //        [[3, 4]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1],[2],[3]]);
            given=  b = np.array([[2],[3],[4]]);
            given=  np.dstack((a,b));
            expected=
                "array([[[1, 2]],\n" +
                "       [[2, 3]],\n" +
                "       [[3, 4]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void hstackTest()
        {
            // >>> a = np.array((1,2,3))
            // >>> b = np.array((2,3,4))
            // >>> np.hstack((a,b))
            // array([1, 2, 3, 2, 3, 4])
            // >>> a = np.array([[1],[2],[3]])
            // >>> b = np.array([[2],[3],[4]])
            // >>> np.hstack((a,b))
            // array([[1, 2],
            //        [2, 3],
            //        [3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array((1,2,3));
            given=  b = np.array((2,3,4));
            given=  np.hstack((a,b));
            expected=
                "array([1, 2, 3, 2, 3, 4])";
            Assert.AreEqual(expected, given.repr);
            given=  a = np.array([[1],[2],[3]]);
            given=  b = np.array([[2],[3],[4]]);
            given=  np.hstack((a,b));
            expected=
                "array([[1, 2],\n" +
                "       [2, 3],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void vstackTest()
        {
            // >>> a = np.array([1, 2, 3])
            // >>> b = np.array([2, 3, 4])
            // >>> np.vstack((a,b))
            // array([[1, 2, 3],
            //        [2, 3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([1, 2, 3]);
            given=  b = np.array([2, 3, 4]);
            given=  np.vstack((a,b));
            expected=
                "array([[1, 2, 3],\n" +
                "       [2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> a = np.array([[1], [2], [3]])
            // >>> b = np.array([[2], [3], [4]])
            // >>> np.vstack((a,b))
            // array([[1],
            //        [2],
            //        [3],
            //        [2],
            //        [3],
            //        [4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1], [2], [3]]);
            given=  b = np.array([[2], [3], [4]]);
            given=  np.vstack((a,b));
            expected=
                "array([[1],\n" +
                "       [2],\n" +
                "       [3],\n" +
                "       [2],\n" +
                "       [3],\n" +
                "       [4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void blockTest()
        {
            // The most common use of this function is to build a block matrix
            
            // >>> A = np.eye(2) * 2
            // >>> B = np.eye(3) * 3
            // >>> np.block([
            // ...     [A,               np.zeros((2, 3))],
            // ...     [np.ones((3, 2)), B               ]
            // ... ])
            // array([[ 2.,  0.,  0.,  0.,  0.],
            //        [ 0.,  2.,  0.,  0.,  0.],
            //        [ 1.,  1.,  3.,  0.,  0.],
            //        [ 1.,  1.,  0.,  3.,  0.],
            //        [ 1.,  1.,  0.,  0.,  3.]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.eye(2) * 2;
            given=  B = np.eye(3) * 3;
            given=  np.block([;
            expected=
                "...     [A,               np.zeros((2, 3))],\n" +
                "...     [np.ones((3, 2)), B               ]\n" +
                "... ])\n" +
                "array([[ 2.,  0.,  0.,  0.,  0.],\n" +
                "       [ 0.,  2.,  0.,  0.,  0.],\n" +
                "       [ 1.,  1.,  3.,  0.,  0.],\n" +
                "       [ 1.,  1.,  0.,  3.,  0.],\n" +
                "       [ 1.,  1.,  0.,  0.,  3.]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // With a list of depth 1, block can be used as hstack
            
            // >>> np.block([1, 2, 3])              # hstack([1, 2, 3])
            // array([1, 2, 3])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.block([1, 2, 3])              # hstack([1, 2, 3]);
            expected=
                "array([1, 2, 3])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> a = np.array([1, 2, 3])
            // >>> b = np.array([2, 3, 4])
            // >>> np.block([a, b, 10])             # hstack([a, b, 10])
            // array([1, 2, 3, 2, 3, 4, 10])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([1, 2, 3]);
            given=  b = np.array([2, 3, 4]);
            given=  np.block([a, b, 10])             # hstack([a, b, 10]);
            expected=
                "array([1, 2, 3, 2, 3, 4, 10])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> A = np.ones((2, 2), int)
            // >>> B = 2 * A
            // >>> np.block([A, B])                 # hstack([A, B])
            // array([[1, 1, 2, 2],
            //        [1, 1, 2, 2]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.ones((2, 2), int);
            given=  B = 2 * A;
            given=  np.block([A, B])                 # hstack([A, B]);
            expected=
                "array([[1, 1, 2, 2],\n" +
                "       [1, 1, 2, 2]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // With a list of depth 2, block can be used in place of vstack:
            
            // >>> a = np.array([1, 2, 3])
            // >>> b = np.array([2, 3, 4])
            // >>> np.block([[a], [b]])             # vstack([a, b])
            // array([[1, 2, 3],
            //        [2, 3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([1, 2, 3]);
            given=  b = np.array([2, 3, 4]);
            given=  np.block([[a], [b]])             # vstack([a, b]);
            expected=
                "array([[1, 2, 3],\n" +
                "       [2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> A = np.ones((2, 2), int)
            // >>> B = 2 * A
            // >>> np.block([[A], [B]])             # vstack([A, B])
            // array([[1, 1],
            //        [1, 1],
            //        [2, 2],
            //        [2, 2]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.ones((2, 2), int);
            given=  B = 2 * A;
            given=  np.block([[A], [B]])             # vstack([A, B]);
            expected=
                "array([[1, 1],\n" +
                "       [1, 1],\n" +
                "       [2, 2],\n" +
                "       [2, 2]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // It can also be used in places of atleast_1d and atleast_2d
            
            // >>> a = np.array(0)
            // >>> b = np.array([1])
            // >>> np.block([a])                    # atleast_1d(a)
            // array([0])
            // >>> np.block([b])                    # atleast_1d(b)
            // array([1])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array(0);
            given=  b = np.array([1]);
            given=  np.block([a])                    # atleast_1d(a);
            expected=
                "array([0])";
            Assert.AreEqual(expected, given.repr);
            given=  np.block([b])                    # atleast_1d(b);
            expected=
                "array([1])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.block([[a]])                  # atleast_2d(a)
            // array([[0]])
            // >>> np.block([[b]])                  # atleast_2d(b)
            // array([[1]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.block([[a]])                  # atleast_2d(a);
            expected=
                "array([[0]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.block([[b]])                  # atleast_2d(b);
            expected=
                "array([[1]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void splitTest()
        {
            // >>> x = np.arange(9.0)
            // >>> np.split(x, 3)
            // [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(9.0);
            given=  np.split(x, 3);
            expected=
                "[array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(8.0)
            // >>> np.split(x, [3, 5, 6, 10])
            // [array([ 0.,  1.,  2.]),
            //  array([ 3.,  4.]),
            //  array([ 5.]),
            //  array([ 6.,  7.]),
            //  array([], dtype=float64)]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(8.0);
            given=  np.split(x, [3, 5, 6, 10]);
            expected=
                "[array([ 0.,  1.,  2.]),\n" +
                " array([ 3.,  4.]),\n" +
                " array([ 5.]),\n" +
                " array([ 6.,  7.]),\n" +
                " array([], dtype=float64)]";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void tileTest()
        {
            // >>> a = np.array([0, 1, 2])
            // >>> np.tile(a, 2)
            // array([0, 1, 2, 0, 1, 2])
            // >>> np.tile(a, (2, 2))
            // array([[0, 1, 2, 0, 1, 2],
            //        [0, 1, 2, 0, 1, 2]])
            // >>> np.tile(a, (2, 1, 2))
            // array([[[0, 1, 2, 0, 1, 2]],
            //        [[0, 1, 2, 0, 1, 2]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([0, 1, 2]);
            given=  np.tile(a, 2);
            expected=
                "array([0, 1, 2, 0, 1, 2])";
            Assert.AreEqual(expected, given.repr);
            given=  np.tile(a, (2, 2));
            expected=
                "array([[0, 1, 2, 0, 1, 2],\n" +
                "       [0, 1, 2, 0, 1, 2]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.tile(a, (2, 1, 2));
            expected=
                "array([[[0, 1, 2, 0, 1, 2]],\n" +
                "       [[0, 1, 2, 0, 1, 2]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> b = np.array([[1, 2], [3, 4]])
            // >>> np.tile(b, 2)
            // array([[1, 2, 1, 2],
            //        [3, 4, 3, 4]])
            // >>> np.tile(b, (2, 1))
            // array([[1, 2],
            //        [3, 4],
            //        [1, 2],
            //        [3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  b = np.array([[1, 2], [3, 4]]);
            given=  np.tile(b, 2);
            expected=
                "array([[1, 2, 1, 2],\n" +
                "       [3, 4, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.tile(b, (2, 1));
            expected=
                "array([[1, 2],\n" +
                "       [3, 4],\n" +
                "       [1, 2],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> c = np.array([1,2,3,4])
            // >>> np.tile(c,(4,1))
            // array([[1, 2, 3, 4],
            //        [1, 2, 3, 4],
            //        [1, 2, 3, 4],
            //        [1, 2, 3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  c = np.array([1,2,3,4]);
            given=  np.tile(c,(4,1));
            expected=
                "array([[1, 2, 3, 4],\n" +
                "       [1, 2, 3, 4],\n" +
                "       [1, 2, 3, 4],\n" +
                "       [1, 2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void repeatTest()
        {
            // >>> np.repeat(3, 4)
            // array([3, 3, 3, 3])
            // >>> x = np.array([[1,2],[3,4]])
            // >>> np.repeat(x, 2)
            // array([1, 1, 2, 2, 3, 3, 4, 4])
            // >>> np.repeat(x, 3, axis=1)
            // array([[1, 1, 1, 2, 2, 2],
            //        [3, 3, 3, 4, 4, 4]])
            // >>> np.repeat(x, [1, 2], axis=0)
            // array([[1, 2],
            //        [3, 4],
            //        [3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.repeat(3, 4);
            expected=
                "array([3, 3, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            given=  x = np.array([[1,2],[3,4]]);
            given=  np.repeat(x, 2);
            expected=
                "array([1, 1, 2, 2, 3, 3, 4, 4])";
            Assert.AreEqual(expected, given.repr);
            given=  np.repeat(x, 3, axis=1);
            expected=
                "array([[1, 1, 1, 2, 2, 2],\n" +
                "       [3, 3, 3, 4, 4, 4]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.repeat(x, [1, 2], axis=0);
            expected=
                "array([[1, 2],\n" +
                "       [3, 4],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void deleteTest()
        {
            // >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
            // >>> arr
            // array([[ 1,  2,  3,  4],
            //        [ 5,  6,  7,  8],
            //        [ 9, 10, 11, 12]])
            // >>> np.delete(arr, 1, 0)
            // array([[ 1,  2,  3,  4],
            //        [ 9, 10, 11, 12]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]);
            given=  arr;
            expected=
                "array([[ 1,  2,  3,  4],\n" +
                "       [ 5,  6,  7,  8],\n" +
                "       [ 9, 10, 11, 12]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.delete(arr, 1, 0);
            expected=
                "array([[ 1,  2,  3,  4],\n" +
                "       [ 9, 10, 11, 12]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.delete(arr, np.s_[::2], 1)
            // array([[ 2,  4],
            //        [ 6,  8],
            //        [10, 12]])
            // >>> np.delete(arr, [1,3,5], None)
            // array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.delete(arr, np.s_[::2], 1);
            expected=
                "array([[ 2,  4],\n" +
                "       [ 6,  8],\n" +
                "       [10, 12]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.delete(arr, [1,3,5], None);
            expected=
                "array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void insertTest()
        {
            // >>> a = np.array([[1, 1], [2, 2], [3, 3]])
            // >>> a
            // array([[1, 1],
            //        [2, 2],
            //        [3, 3]])
            // >>> np.insert(a, 1, 5)
            // array([1, 5, 1, 2, 2, 3, 3])
            // >>> np.insert(a, 1, 5, axis=1)
            // array([[1, 5, 1],
            //        [2, 5, 2],
            //        [3, 5, 3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1, 1], [2, 2], [3, 3]]);
            given=  a;
            expected=
                "array([[1, 1],\n" +
                "       [2, 2],\n" +
                "       [3, 3]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.insert(a, 1, 5);
            expected=
                "array([1, 5, 1, 2, 2, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            given=  np.insert(a, 1, 5, axis=1);
            expected=
                "array([[1, 5, 1],\n" +
                "       [2, 5, 2],\n" +
                "       [3, 5, 3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Difference between sequence and scalars:
            
            // >>> np.insert(a, [1], [[1],[2],[3]], axis=1)
            // array([[1, 1, 1],
            //        [2, 2, 2],
            //        [3, 3, 3]])
            // >>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),
            // ...                np.insert(a, [1], [[1],[2],[3]], axis=1))
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.insert(a, [1], [[1],[2],[3]], axis=1);
            expected=
                "array([[1, 1, 1],\n" +
                "       [2, 2, 2],\n" +
                "       [3, 3, 3]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),;
            expected=
                "...                np.insert(a, [1], [[1],[2],[3]], axis=1))\n" +
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> b = a.flatten()
            // >>> b
            // array([1, 1, 2, 2, 3, 3])
            // >>> np.insert(b, [2, 2], [5, 6])
            // array([1, 1, 5, 6, 2, 2, 3, 3])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  b = a.flatten();
            given=  b;
            expected=
                "array([1, 1, 2, 2, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            given=  np.insert(b, [2, 2], [5, 6]);
            expected=
                "array([1, 1, 5, 6, 2, 2, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.insert(b, slice(2, 4), [5, 6])
            // array([1, 1, 5, 2, 6, 2, 3, 3])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.insert(b, slice(2, 4), [5, 6]);
            expected=
                "array([1, 1, 5, 2, 6, 2, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.insert(b, [2, 2], [7.13, False]) # type casting
            // array([1, 1, 7, 0, 2, 2, 3, 3])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.insert(b, [2, 2], [7.13, False]) # type casting;
            expected=
                "array([1, 1, 7, 0, 2, 2, 3, 3])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x = np.arange(8).reshape(2, 4)
            // >>> idx = (1, 3)
            // >>> np.insert(x, idx, 999, axis=1)
            // array([[  0, 999,   1,   2, 999,   3],
            //        [  4, 999,   5,   6, 999,   7]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(8).reshape(2, 4);
            given=  idx = (1, 3);
            given=  np.insert(x, idx, 999, axis=1);
            expected=
                "array([[  0, 999,   1,   2, 999,   3],\n" +
                "       [  4, 999,   5,   6, 999,   7]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void appendTest()
        {
            // >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
            // array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]);
            expected=
                "array([1, 2, 3, 4, 5, 6, 7, 8, 9])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // When axis is specified, values must have the correct shape.
            
            // >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
            // array([[1, 2, 3],
            //        [4, 5, 6],
            //        [7, 8, 9]])
            // >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
            // Traceback (most recent call last):
            // ...
            // ValueError: arrays must have same number of dimensions
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0);
            expected=
                "array([[1, 2, 3],\n" +
                "       [4, 5, 6],\n" +
                "       [7, 8, 9]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0);
            expected=
                "Traceback (most recent call last):\n" +
                "...\n" +
                "ValueError: arrays must have same number of dimensions";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void resizeTest()
        {
            // >>> a=np.array([[0,1],[2,3]])
            // >>> np.resize(a,(2,3))
            // array([[0, 1, 2],
            //        [3, 0, 1]])
            // >>> np.resize(a,(1,4))
            // array([[0, 1, 2, 3]])
            // >>> np.resize(a,(2,4))
            // array([[0, 1, 2, 3],
            //        [0, 1, 2, 3]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a=np.array([[0,1],[2,3]]);
            given=  np.resize(a,(2,3));
            expected=
                "array([[0, 1, 2],\n" +
                "       [3, 0, 1]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.resize(a,(1,4));
            expected=
                "array([[0, 1, 2, 3]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.resize(a,(2,4));
            expected=
                "array([[0, 1, 2, 3],\n" +
                "       [0, 1, 2, 3]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void trim_zerosTest()
        {
            // >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
            // >>> np.trim_zeros(a)
            // array([1, 2, 3, 0, 2, 1])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0));
            given=  np.trim_zeros(a);
            expected=
                "array([1, 2, 3, 0, 2, 1])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.trim_zeros(a, 'b')
            // array([0, 0, 0, 1, 2, 3, 0, 2, 1])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.trim_zeros(a, 'b');
            expected=
                "array([0, 0, 0, 1, 2, 3, 0, 2, 1])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // The input data type is preserved, list/tuple in means list/tuple out.
            
            // >>> np.trim_zeros([0, 1, 2, 0])
            // [1, 2]
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.trim_zeros([0, 1, 2, 0]);
            expected=
                "[1, 2]";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void uniqueTest()
        {
            // >>> np.unique([1, 1, 2, 2, 3, 3])
            // array([1, 2, 3])
            // >>> a = np.array([[1, 1], [2, 3]])
            // >>> np.unique(a)
            // array([1, 2, 3])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.unique([1, 1, 2, 2, 3, 3]);
            expected=
                "array([1, 2, 3])";
            Assert.AreEqual(expected, given.repr);
            given=  a = np.array([[1, 1], [2, 3]]);
            given=  np.unique(a);
            expected=
                "array([1, 2, 3])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Return the unique rows of a 2D array
            
            // >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
            // >>> np.unique(a, axis=0)
            // array([[1, 0, 0], [2, 3, 4]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]]);
            given=  np.unique(a, axis=0);
            expected=
                "array([[1, 0, 0], [2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Return the indices of the original array that give the unique values:
            
            // >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
            // >>> u, indices = np.unique(a, return_index=True)
            // >>> u
            // array(['a', 'b', 'c'],
            //        dtype='|S1')
            // >>> indices
            // array([0, 1, 3])
            // >>> a[indices]
            // array(['a', 'b', 'c'],
            //        dtype='|S1')
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array(['a', 'b', 'b', 'c', 'a']);
            given=  u, indices = np.unique(a, return_index=True);
            given=  u;
            expected=
                "array(['a', 'b', 'c'],\n" +
                "       dtype='|S1')";
            Assert.AreEqual(expected, given.repr);
            given=  indices;
            expected=
                "array([0, 1, 3])";
            Assert.AreEqual(expected, given.repr);
            given=  a[indices];
            expected=
                "array(['a', 'b', 'c'],\n" +
                "       dtype='|S1')";
            Assert.AreEqual(expected, given.repr);
            #endif
            // Reconstruct the input array from the unique values:
            
            // >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
            // >>> u, indices = np.unique(a, return_inverse=True)
            // >>> u
            // array([1, 2, 3, 4, 6])
            // >>> indices
            // array([0, 1, 4, 3, 1, 2, 1])
            // >>> u[indices]
            // array([1, 2, 6, 4, 2, 3, 2])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  a = np.array([1, 2, 6, 4, 2, 3, 2]);
            given=  u, indices = np.unique(a, return_inverse=True);
            given=  u;
            expected=
                "array([1, 2, 3, 4, 6])";
            Assert.AreEqual(expected, given.repr);
            given=  indices;
            expected=
                "array([0, 1, 4, 3, 1, 2, 1])";
            Assert.AreEqual(expected, given.repr);
            given=  u[indices];
            expected=
                "array([1, 2, 6, 4, 2, 3, 2])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void flipTest()
        {
            // >>> A = np.arange(8).reshape((2,2,2))
            // >>> A
            // array([[[0, 1],
            //         [2, 3]],
            //        [[4, 5],
            //         [6, 7]]])
            // >>> flip(A, 0)
            // array([[[4, 5],
            //         [6, 7]],
            //        [[0, 1],
            //         [2, 3]]])
            // >>> flip(A, 1)
            // array([[[2, 3],
            //         [0, 1]],
            //        [[6, 7],
            //         [4, 5]]])
            // >>> np.flip(A)
            // array([[[7, 6],
            //         [5, 4]],
            //        [[3, 2],
            //         [1, 0]]])
            // >>> np.flip(A, (0, 2))
            // array([[[5, 4],
            //         [7, 6]],
            //        [[1, 0],
            //         [3, 2]]])
            // >>> A = np.random.randn(3,4,5)
            // >>> np.all(flip(A,2) == A[:,:,::-1,...])
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.arange(8).reshape((2,2,2));
            given=  A;
            expected=
                "array([[[0, 1],\n" +
                "        [2, 3]],\n" +
                "       [[4, 5],\n" +
                "        [6, 7]]])";
            Assert.AreEqual(expected, given.repr);
            given=  flip(A, 0);
            expected=
                "array([[[4, 5],\n" +
                "        [6, 7]],\n" +
                "       [[0, 1],\n" +
                "        [2, 3]]])";
            Assert.AreEqual(expected, given.repr);
            given=  flip(A, 1);
            expected=
                "array([[[2, 3],\n" +
                "        [0, 1]],\n" +
                "       [[6, 7],\n" +
                "        [4, 5]]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.flip(A);
            expected=
                "array([[[7, 6],\n" +
                "        [5, 4]],\n" +
                "       [[3, 2],\n" +
                "        [1, 0]]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.flip(A, (0, 2));
            expected=
                "array([[[5, 4],\n" +
                "        [7, 6]],\n" +
                "       [[1, 0],\n" +
                "        [3, 2]]])";
            Assert.AreEqual(expected, given.repr);
            given=  A = np.random.randn(3,4,5);
            given=  np.all(flip(A,2) == A[:,:,::-1,...]);
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void fliplrTest()
        {
            // >>> A = np.diag([1.,2.,3.])
            // >>> A
            // array([[ 1.,  0.,  0.],
            //        [ 0.,  2.,  0.],
            //        [ 0.,  0.,  3.]])
            // >>> np.fliplr(A)
            // array([[ 0.,  0.,  1.],
            //        [ 0.,  2.,  0.],
            //        [ 3.,  0.,  0.]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.diag([1.,2.,3.]);
            given=  A;
            expected=
                "array([[ 1.,  0.,  0.],\n" +
                "       [ 0.,  2.,  0.],\n" +
                "       [ 0.,  0.,  3.]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.fliplr(A);
            expected=
                "array([[ 0.,  0.,  1.],\n" +
                "       [ 0.,  2.,  0.],\n" +
                "       [ 3.,  0.,  0.]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> A = np.random.randn(2,3,5)
            // >>> np.all(np.fliplr(A) == A[:,::-1,...])
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.random.randn(2,3,5);
            given=  np.all(np.fliplr(A) == A[:,::-1,...]);
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void flipudTest()
        {
            // >>> A = np.diag([1.0, 2, 3])
            // >>> A
            // array([[ 1.,  0.,  0.],
            //        [ 0.,  2.,  0.],
            //        [ 0.,  0.,  3.]])
            // >>> np.flipud(A)
            // array([[ 0.,  0.,  3.],
            //        [ 0.,  2.,  0.],
            //        [ 1.,  0.,  0.]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.diag([1.0, 2, 3]);
            given=  A;
            expected=
                "array([[ 1.,  0.,  0.],\n" +
                "       [ 0.,  2.,  0.],\n" +
                "       [ 0.,  0.,  3.]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.flipud(A);
            expected=
                "array([[ 0.,  0.,  3.],\n" +
                "       [ 0.,  2.,  0.],\n" +
                "       [ 1.,  0.,  0.]])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> A = np.random.randn(2,3,5)
            // >>> np.all(np.flipud(A) == A[::-1,...])
            // True
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  A = np.random.randn(2,3,5);
            given=  np.all(np.flipud(A) == A[::-1,...]);
            expected=
                "True";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> np.flipud([1,2])
            // array([2, 1])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  np.flipud([1,2]);
            expected=
                "array([2, 1])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void rollTest()
        {
            // >>> x = np.arange(10)
            // >>> np.roll(x, 2)
            // array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x = np.arange(10);
            given=  np.roll(x, 2);
            expected=
                "array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])";
            Assert.AreEqual(expected, given.repr);
            #endif
            // >>> x2 = np.reshape(x, (2,5))
            // >>> x2
            // array([[0, 1, 2, 3, 4],
            //        [5, 6, 7, 8, 9]])
            // >>> np.roll(x2, 1)
            // array([[9, 0, 1, 2, 3],
            //        [4, 5, 6, 7, 8]])
            // >>> np.roll(x2, 1, axis=0)
            // array([[5, 6, 7, 8, 9],
            //        [0, 1, 2, 3, 4]])
            // >>> np.roll(x2, 1, axis=1)
            // array([[4, 0, 1, 2, 3],
            //        [9, 5, 6, 7, 8]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  x2 = np.reshape(x, (2,5));
            given=  x2;
            expected=
                "array([[0, 1, 2, 3, 4],\n" +
                "       [5, 6, 7, 8, 9]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.roll(x2, 1);
            expected=
                "array([[9, 0, 1, 2, 3],\n" +
                "       [4, 5, 6, 7, 8]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.roll(x2, 1, axis=0);
            expected=
                "array([[5, 6, 7, 8, 9],\n" +
                "       [0, 1, 2, 3, 4]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.roll(x2, 1, axis=1);
            expected=
                "array([[4, 0, 1, 2, 3],\n" +
                "       [9, 5, 6, 7, 8]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
        [TestMethod]
        public void rot90Test()
        {
            // >>> m = np.array([[1,2],[3,4]], int)
            // >>> m
            // array([[1, 2],
            //        [3, 4]])
            // >>> np.rot90(m)
            // array([[2, 4],
            //        [1, 3]])
            // >>> np.rot90(m, 2)
            // array([[4, 3],
            //        [2, 1]])
            // >>> m = np.arange(8).reshape((2,2,2))
            // >>> np.rot90(m, 1, (1,2))
            // array([[[1, 3],
            //         [0, 2]],
            //        [[5, 7],
            //         [4, 6]]])
            // 
            
            #if TODO
            object given = null;
            object expected = null;
            given=  m = np.array([[1,2],[3,4]], int);
            given=  m;
            expected=
                "array([[1, 2],\n" +
                "       [3, 4]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.rot90(m);
            expected=
                "array([[2, 4],\n" +
                "       [1, 3]])";
            Assert.AreEqual(expected, given.repr);
            given=  np.rot90(m, 2);
            expected=
                "array([[4, 3],\n" +
                "       [2, 1]])";
            Assert.AreEqual(expected, given.repr);
            given=  m = np.arange(8).reshape((2,2,2));
            given=  np.rot90(m, 1, (1,2));
            expected=
                "array([[[1, 3],\n" +
                "        [0, 2]],\n" +
                "       [[5, 7],\n" +
                "        [4, 6]]])";
            Assert.AreEqual(expected, given.repr);
            #endif
        }
    }
}
