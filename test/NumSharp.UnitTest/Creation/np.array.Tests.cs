using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_array_tests
    {
        [TestMethod]
        public void nparray_1d()
        {
            var v = np.array(new int[] {1, 2, 3, 4, 5, 6, 7, 8});
            v.shape.Should().ContainInOrder(8);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_params()
        {
            var v = np.array(1, 2, 3, 4, 5, 6, 7, 8);

            v.shape.Should().ContainInOrder(8);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_1d_typeless()
        {
            var v = np.array((Array)new int[] {1, 2, 3, 4, 5, 6, 7, 8});

            v.shape.Should().ContainInOrder(8);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_3d_jagged()
        {
            var v = np.array((Array)new int[,] {{1, 2, 3, 4}, {5, 6, 7, 8}});

            v.shape.Should().ContainInOrder(2,4);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_3d_typeless()
        {
            // @formatter:off — disable formatter after this line
            var v = np.array((Array) new int[][][]
            {
                new int[][]
                {
                    new int[] {1, 2}, new int[] {3, 4}
                }, new int[][]
                {
                    new int[] {5, 6}, new int[] {7, 8}
                }
            });
            // @formatter:on — disable formatter after this line

            v.shape.Should().ContainInOrder(2, 2, 2);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_2d()
        {
            var v = np.array(new int[][] {new int[] {1, 2, 3, 4}, new int[] {5, 6, 7, 8}});

            v.shape.Should().ContainInOrder(2, 4);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_3d()
        {
            // @formatter:off — disable formatter after this line
            var v = np.array(new int[][][]
            {
                new int[][]
                {
                    new int[] {1, 2}, new int[] {3, 4}
                }, new int[][]
                {
                    new int[] {5, 6}, new int[] {7, 8}
                }
            });
            // @formatter:on — disable formatter after this line

            v.shape.Should().ContainInOrder(2, 2, 2);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_4d()
        {
            // @formatter:off — disable formatter after this line
            var v = np.array(new int[][][][] {new int[][][] {
                new int[][]
            {
                new int[]{1},
                new int[]{2},
                new int[]{3},
                new int[]{4},
            },           
                new int[][]
            {            
                new int[]{5},
                new int[]{6},
                new int[]{7},
                new int[]{8},
            },}});
            // @formatter:on — disable formatter after this line

            v.shape.Should().ContainInOrder(1, 2, 4, 1);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void dev()
        {
            var a = new int[] {1, 2, 3};
            var b = new int[] {4, 5, 6};
            var r = Arrays.Concat(a, b);
        }
    }
}
