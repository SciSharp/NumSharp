using System;
using NumSharp.Tests.Utilities;
using NumSharp.Utilities;

namespace NumSharp.Tests.Creation
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

            v.shape.Should().ContainInOrder(2, 4);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_1d_typeless_knowntype()
        {
            var v = np.array(new int[] {1, 2, 3, 4, 5, 6, 7, 8});

            v.shape.Should().ContainInOrder(8);
            v.size.Should().Be(8);
        }

        [TestMethod]
        public void nparray_3d_jagged_knowntype()
        {
            var v = np.array(new int[,] {{1, 2, 3, 4}, {5, 6, 7, 8}});

            v.shape.Should().ContainInOrder(2, 4);
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
        public void Arrays_Concat()
        {
            var a = new int[] {1, 2, 3};
            var b = new int[] {4, 5, 6};
            var r = Arrays.Concat(a, b);
            r.Should().ContainInOrder(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Array_Copy()
        {
            var arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            np.array(arr, copy: true).Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            np.array(arr, copy: false).Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Array_NonCopy()
        {
            var arr = new int[,] {{1, 2, 3}, {4, 5, 6}};
            var nd = np.array(arr, copy: false);
            nd.Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            arr[0, 2] = 0;
            nd[0, 2].Should().BeOfValues(0);
        }

        [TestMethod]
        public void Array_Jagged()
        {
            var arr = new int[][] {new int[] {1, 2, 3}, new int[] {4, 5, 6}};
            var nd = np.array(arr);
            nd.Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            arr[0][2] = 0;
            nd[0, 2].Should().BeOfValues(3);
        }

        /// <summary>
        ///     Parity QA-35 (fixed 2026-11-05; this pin came with the ufunc result-layout port, 2026-11-08):
        ///     <c>np.array(a)</c> is a copy in NumPy's defaults <c>copy=True, order='K'</c>, and a copy NumPy allocates is
        ///     always writeable — so
        ///     <c>np.array(ro)[0, 0] = 99</c> works on a read-only <c>ro</c>, and the copy of
        ///     <c>broadcast_to(arange(4.), (3, 4))</c> (strides <c>(0, 8)</c>) is F-contiguous (the stride-0 axis sorts
        ///     fastest). The storage clone used to keep a read-only source's flag and lay a broadcast out in C order.
        /// </summary>
        [TestMethod]
        public void ArrayOfNDArray_IsAWriteableKOrderCopy()
        {
            using var ro = np.arange(6.0).reshape(2, 3);
            ro.setflags(write: false);
            using var bcast = np.broadcast_to(np.arange(4.0), new Shape(3, 4));
            using var permuted = np.arange(24.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

            using var copy = np.array(ro);
            using var bcastCopy = np.array(bcast);
            using var permutedCopy = np.array(permuted);
            Action write = () => copy[0, 0] = (NDArray)99.0;

            copy.flags.writeable.Should().BeTrue("a copy NumPy allocates is writeable");
            write.Should().NotThrow();
            ro.GetDouble(0, 0).Should().Be(0.0);
            bcastCopy.flags.f_contiguous.Should().BeTrue("order='K' keeps the stride-0 axis fastest");
            bcastCopy.flags.c_contiguous.Should().BeFalse();
            // np.array(a.transpose(1, 0, 2)).strides == (32, 64, 8) in NumPy 2.4.2: the K copy keeps the transpose.
            permutedCopy.strides.Should().Equal(32L, 64L, 8L);
            permutedCopy.flags.owndata.Should().BeTrue();
        }
    }
}
