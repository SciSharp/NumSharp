using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using NumSharp.Extensions;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    public class ShapeTest
    {
        [Test]
        public void Index()
        {
            var shape0 = new Shape(4, 3);

            int idx0 = shape0.GetOffset(2, 1);

            Assert.IsTrue(idx0 == 3 * 2 + 1 * 1);
        }

        [Test]
        public void CheckIndexing()
        {
            var shape0 = new Shape(4, 3, 2);

            int[] strgDimSize = shape0.Strides;

            int index = shape0.GetOffset(1, 2, 1);

            Assert.IsTrue(shape0.GetCoordinates(index).SequenceEqual(new int[] { 1, 2, 1 }));

            var rnd = new Randomizer();
            var randomIndex = new int[] { rnd.Next(0, 3), rnd.Next(0, 2), rnd.Next(0, 1) };

            int index1 = shape0.GetOffset(randomIndex);
            Assert.IsTrue(shape0.GetCoordinates(index1).SequenceEqual(randomIndex));

            var shape1 = new Shape(2, 3, 4);

            index = shape1.GetOffset(1, 2, 1);
            Assert.IsTrue(shape1.GetCoordinates(index).SequenceEqual(new int[] { 1, 2, 1 }));

            randomIndex = new int[] { rnd.Next(0, 1), rnd.Next(0, 2), rnd.Next(0, 3) };
            index = shape1.GetOffset(randomIndex);
            Assert.IsTrue(shape1.GetCoordinates(index).SequenceEqual(randomIndex));

            randomIndex = new int[] { rnd.Next(1, 10), rnd.Next(1, 10), rnd.Next(1, 10) };

            var shape2 = new Shape(randomIndex);

            randomIndex = new int[] { rnd.Next(0, shape2.Dimensions[0]), rnd.Next(0, shape2.Dimensions[1]), rnd.Next(0, shape2.Dimensions[2]) };

            index = shape2.GetOffset(randomIndex);
            Assert.IsTrue(shape2.GetCoordinates(index).SequenceEqual(randomIndex));
        }

        [Test, Skip("Ignored")]
        public void CheckColRowSwitch()
        {
            var shape1 = new Shape(5);
            Assert.IsTrue(shape1.Strides.SequenceEqual(new int[] { 1 }));

            shape1.ChangeTensorLayout();
            Assert.IsTrue(shape1.Strides.SequenceEqual(new int[] { 1 }));

            var shape2 = new Shape(4, 3);
            Assert.IsTrue(shape2.Strides.SequenceEqual(new int[] { 1, 4 }));

            shape2.ChangeTensorLayout();
            Assert.IsTrue(shape2.Strides.SequenceEqual(new int[] { 3, 1 }));

            var shape3 = new Shape(2, 3, 4);
            Assert.IsTrue(shape3.Strides.SequenceEqual(new int[] { 1, 2, 6 }));

            shape3.ChangeTensorLayout();
            Assert.IsTrue(shape3.Strides.SequenceEqual(new int[] { 12, 4, 1 }));
        }

        /// <summary>
        ///     Based on issue https://github.com/SciSharp/NumSharp/issues/306
        /// </summary>
        [Test]
        public void EqualityComparer()
        {
            Shape a = null;
            Shape b = null;

            (a == b).Should().BeTrue();
            (a == null).Should().BeTrue();
            (null == b).Should().BeTrue();

            a = (Shape)5;
            b = (Shape)4;
            (a != b).Should().BeTrue();

            b = (Shape)5;
            (a == b).Should().BeTrue();

            a = new Shape(1, 2, 3, 4, 5);
            b = new Shape(1, 2, 3, 4, 5);
            (a == b).Should().BeTrue();
            b = new Shape(1, 2, 3, 4);
            (a != b).Should().BeTrue();
        }


        [Test, TUnit.Core.Timeout(10000)]
        public void ExtractShape_FromArray(CancellationToken cancellationToken)
        {
            // @formatter:off — disable formatter after this line
            var v = Shape.ExtractShape((Array)new int[][][]
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

            v.Should().ContainInOrder(2, 2, 2);
            v.Sum().Should().Be(6);

            v = Shape.ExtractShape(new int[][] { new int[] { 1, 2, 3, 4 }, new int[] { 5, 6, 7, 8 } });

            v.Should().ContainInOrder(2, 4);
            v.Sum().Should().Be(6);

            // @formatter:off — disable formatter after this line
            v = Shape.ExtractShape(new int[][][]
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

            v.Should().ContainInOrder(2, 2, 2);
            v.Sum().Should().Be(6);

            var jagged = new int[5, 3, 2];
            Shape.ExtractShape(jagged).Should().ContainInOrder(5, 3, 2);
            Shape.ExtractShape(new int[5]).Should().ContainInOrder(5);
        }

        [Test]
        public void Create_Vector()
        {
            Shape.Vector(10).Should().Be(new Shape(10));
            Shape.Vector(10).strides.Should().ContainInOrder(new Shape(10).strides);

            Shape.Vector(1).Should().Be(new Shape(1));
            Shape.Vector(1).strides.Should().ContainInOrder(new Shape(1).strides);

            Shape.Vector(0).Should().Be(new Shape(0));
            Shape.Vector(0).strides.Should().ContainInOrder(new Shape(0).strides);
        }

        [Test]
        public void Create_Matrix()
        {
            Shape.Matrix(5, 5).Should().Be(new Shape(5, 5));
            Shape.Matrix(5, 5).strides.Should().ContainInOrder(new Shape(5, 5).strides);

            Shape.Matrix(1, 5).Should().Be(new Shape(1, 5));
            Shape.Matrix(1, 5).strides.Should().ContainInOrder(new Shape(1, 5).strides);

            Shape.Matrix(5, 1).Should().Be(new Shape(5, 1));
            Shape.Matrix(5, 1).strides.Should().ContainInOrder(new Shape(5, 1).strides);

            Shape.Matrix(5, 0).Should().Be(new Shape(5, 0));
            Shape.Matrix(5, 0).strides.Should().ContainInOrder(new Shape(5, 0).strides);

            Shape.Matrix(0, 0).Should().Be(new Shape(0, 0));
            Shape.Matrix(0, 0).strides.Should().ContainInOrder(new Shape(0, 0).strides);
        }

        [Test]
        public void GetAxis()
        {
            var baseshape = new Shape(2, 3, 4, 5);
            Shape.GetAxis(baseshape, 0).Should().ContainInOrder(3, 4, 5);
            Shape.GetAxis(baseshape, 1).Should().ContainInOrder(2, 4, 5);
            Shape.GetAxis(baseshape, 2).Should().ContainInOrder(2, 3, 5);
            Shape.GetAxis(baseshape, 3).Should().ContainInOrder(2, 3, 4);
            Shape.GetAxis(baseshape, -1).Should().ContainInOrder(2, 3, 4);
        }

        [Test]
        public void GetSubshape()
        {
            //initialize
            (Shape Shape, int Offset) ret;
            var nd = new NDArray(new ArraySlice<int>(new UnmanagedMemoryBlock<int>(25, 0)), new Shape(5, 5));
            var arr = new int[5, 5];
            var arr2 = new int[5, 1, 5];

            for (int i = 0; i < nd.size; i++)
            {
                nd.Storage.SetAtIndex(i, i);
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    arr[i, j] = i * 5 + j;
                }
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    for (int k = 0; k < 5; k++)
                    {
                        arr2[i, j, k] = i * 5 + j * 1 + k;
                    }
                }
            }


            //test case 1
            nd.Shape = new Shape(5, 5);

            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(0);
            arr[0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5);
            arr[1, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 4);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 4);
            arr[1, 4].Should().Be(ret.Offset);


            //test case 2
            nd.Shape = new Shape(5, 1, 5);
            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(0);
            arr2[0, 0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(5);
            arr2[1, 0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0, 1);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 1);
            arr2[1, 0, 1].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(2, 0, 1);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 * 2 + 1);
            arr2[2, 0, 1].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(0);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(5);

            ret = nd.Shape.GetSubshape(1, 0, 3);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 3);
            arr2[1, 0, 3].Should().Be(ret.Offset);


            //test case 3
            nd.Shape = new Shape(1, 1, 5, 5);
            ret = nd.Shape.GetSubshape(0, 0, 3, 3);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(18);

            ret = nd.Shape.GetSubshape(0, 0, 3);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(15);

            ret = ret.Shape.GetSubshape(2);
            ret.Shape.Size.Should().Be(1);
            ret.Shape.NDim.Should().Be(0);
            ret.Shape.IsScalar.Should().BeTrue();


            //test case 4
            nd.Shape = new Shape(1, 5, 5, 1);
            ret = nd.Shape.GetSubshape(0, 1);
            ret.Offset.Should().Be(5);
            ret.Shape.NDim.Should().Be(2);
            ret.Shape.Dimensions[0].Should().Be(5);
            ret.Shape.Dimensions[1].Should().Be(1);
        }

        [Test]
        public void ShapeSlicing_1D()
        {
            new Shape(10).Slice(":").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>1*10)"));
            new Shape(10).Slice("-77:77").Should().Be(new Shape(10));
            new Shape(10).Slice("-77:77").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>1*10)"));
            new Shape(10).Slice(":7").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>1*7)"));
            new Shape(10).Slice("7:").ViewInfo.Slices[0].Should().Be(new SliceDef("(7>>1*3)"));
            new Shape(10).Slice("-7:").ViewInfo.Slices[0].Should().Be(new SliceDef("(3>>1*7)"));
        }

        [Test]
        public void RepeatedSlicing_1D()
        {
            new Shape(10).Slice(":").Slice(":").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>1*10)"));
            new Shape(10).Slice(":5").Slice("2:").ViewInfo.Slices[0].Should().Be(new SliceDef("(2>>1*3)"));
            new Shape(10).Slice(":5").Slice("2:").Slice("::2").ViewInfo.Slices[0].Should().Be(new SliceDef("(2>>2*2)"));
            new Shape(10).Slice("1:9").Slice("::-2").ViewInfo.Slices[0].Should().Be(new SliceDef("(8>>-2*4)"));
            new Shape(10).Slice("1:9").Slice("::2").ViewInfo.Slices[0].Should().Be(new SliceDef("(1>>2*4)"));
            new Shape(10).Slice("9:2:-2").ViewInfo.Slices[0].Should().Be(new SliceDef("(9>>-2*4)"));
            new Shape(10).Slice("9:2:-2").Slice("::-3").ViewInfo.Slices[0].Should().Be(new SliceDef("(3>>6*2)"));
            new Shape(10).Slice("1:9").Slice("::-2").Slice("::-3").ViewInfo.Slices[0].Should().Be(new SliceDef("(2>>6*2)"));
            new Shape(10).Slice("9:2:-2").Slice("::2").ViewInfo.Slices[0].Should().Be(new SliceDef("(9>>-4*2)"));
            new Shape(10).Slice("9:2:-2").Slice("::-2").ViewInfo.Slices[0].Should().Be(new SliceDef("(3>>4*2)"));
            new Shape(10).Slice("1:9").Slice("::-2").Slice("::-2").ViewInfo.Slices[0].Should().Be(new SliceDef("(2>>4*2)"));
            new Shape(10).Slice("0:9").Slice("::-2").Slice("::-2").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>4*3)"));
            new Shape(20).Slice("3:19").Slice("1:15:2").Slice("2:6:2").ViewInfo.Slices[0].Should().Be(new SliceDef("(8>>4*2)"));
            // the repeatedly sliced shape needs to have the original shape
            new Shape(20).Slice("3:19").Slice("1:15:2").Slice("2:6:2").ViewInfo.OriginalShape.Should().Be(new Shape(20));
        }

        [Test]
        public void ShapeSlicing_2D()
        {
            new Shape(3, 3).Slice(":,1:").Should().Be(new Shape(3, 2));
            new Shape(3, 3).Slice(":,1:").ViewInfo.Slices[0].Should().Be(new SliceDef("(0>>1*3)"));
            new Shape(3, 3).Slice(":,1:").ViewInfo.Slices[1].Should().Be(new SliceDef("(1>>1*2)"));
        }


        [Test]
        public void GetCoordsFromIndex_2D()
        {
            var shape = new Shape(3, 3).Slice(":,1:");
            // todo: test get coords from index with sliced shapes
        }

        [Test]
        public void ExpandDim_Case1()
        {
            Shape shape = (3, 3, 3);
            shape = shape.ExpandDimension(1);
            Console.WriteLine(shape);
            shape.GetOffset(2, 0, 0, 2).Should().Be(9 * 2 + 2);
        }

        [Test]
        public void ExpandDim_Case2()
        {
            Shape shape = (3, 3, 3);
            shape = shape.ExpandDimension(0);
            shape.GetOffset(0, 2, 0, 2).Should().Be(9 * 2 + 2);
        }

        [Test]
        public void ExpandDim_Case3()
        {
            Shape shape = (3, 3, 3);
            shape = shape.ExpandDimension(2);
            Console.WriteLine(shape);
            shape.GetOffset(2, 0, 0, 2).Should().Be(9 * 2 + 2);
        }

        [Test]
        public void ExpandDim_Case4()
        {
            Shape shape = (3, 3, 3);
            shape = shape.ExpandDimension(3);
            Console.WriteLine(shape);
            shape.GetOffset(2, 0, 2, 0).Should().Be(9 * 2 + 2);
        }


        [Test]
        public void ExpandDim0_Slice()
        {
            //>>> a = np.arange(27).reshape(3, 3, 3)[0, :]
            //>>> a
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> a.shape
            //(3L, 3L)
            //>>> b=a.reshape(3,1,3)
            //>>> b
            //array([[[0, 1, 2]],
            //
            //       [[3, 4, 5]],
            //
            //       [[6, 7, 8]]])
            //>>> b.shape
            //(3L, 1L, 3L)
            //>>> b[2, 0, 2]
            //8
            Shape shape = (3, 3, 3);
            shape = shape.Slice("0,:");
            shape = shape.ExpandDimension(1);
            Console.WriteLine(shape);
            shape.GetOffset(2, 0, 2).Should().Be(8);
            shape.Should().Be(new Shape(3, 1, 3));
        }

        [Test]
        public void ExpandDim1_Slice()
        {
            //>>> a = np.arange(3 * 2 * 3).reshape(3, 2, 3)[1, :]
            //>>> a
            //array([[6, 7, 8],
            //       [ 9, 10, 11]])
            //>>> a.shape
            //(2L, 3L)
            //>>> b=a.reshape(2,1,3)
            //>>> b
            //array([[[ 6,  7,  8]],

            //       [[ 9, 10, 11]]])
            //>>> b[0, 0, 2]
            //8
            Shape shape = (3, 2, 3);
            shape = shape.Slice("1,:");
            shape = shape.ExpandDimension(1);
            shape.GetOffset(0, 0, 2).Should().Be(8);
            shape.Should().Be(new Shape(2, 1, 3));
        }

        //[Test]
        //public void Strides_Case1()
        //{
        //    var a = np.arange(3 * 2 * 2).reshape((3, 2, 2));

        //    Console.WriteLine(a.strides.ToString(false));
        //    a.Shape.Strides.Should().BeEquivalentTo(new int[] {16, 8, 4});
        //}

        [Test]
        public void HashcodeComputation()
        {
            var a = Shape.Vector(5);
            var b = new Shape(5);
            a._hashCode.Should().Be(b._hashCode);
            a.ComputeHashcode();
            a._hashCode.Should().Be(b._hashCode);

            a = Shape.Matrix(5, 10);
            b = new Shape(5, 10);
            a._hashCode.Should().Be(b._hashCode);
            a.ComputeHashcode();
            a._hashCode.Should().Be(b._hashCode);

            a = new Shape(3, 3, 3);
            b = new Shape(3, 3, 3);
            a._hashCode.Should().Be(b._hashCode);
            b.ComputeHashcode();
            a._hashCode.Should().Be(b._hashCode);
            a.ComputeHashcode();
            a._hashCode.Should().Be(b._hashCode);
        }

        [Test]
        public void HashcodeScalars()
        {
            Shape.Scalar.GetHashCode().Should().Be(int.MinValue);
            Shape.NewScalar().GetHashCode().Should().Be(int.MinValue);
            Shape.NewScalar(new ViewInfo() { OriginalShape = new Shape(1, 2, 3) }).GetHashCode().Should().Be(int.MinValue);
            Shape.NewScalar(new ViewInfo() { OriginalShape = new Shape(1, 2, 3) }, new BroadcastInfo(Shape.Empty(1))).GetHashCode().Should().Be(int.MinValue);
        }

        #region GetCoordinatesFromAbsoluteIndex

        [Test]
        public void GetCoordinatesFromAbsoluteIndex_Unsliced()
        {
            var shape = new Shape(3, 3);
            //[[0 1 2]
            // [3 4 5]
            // [6 7 8]]
            shape.GetOffset(1, 0).Should().Be(3);
            shape.GetCoordinatesFromAbsoluteIndex(3).Should().Equal(new int[] { 1, 0 });
            shape.GetOffset(2, 2).Should().Be(8);
            shape.GetCoordinatesFromAbsoluteIndex(8).Should().Equal(new int[] { 2, 2 });
        }

        [Test]
        public void GetCoordinatesFromAbsoluteIndex_Sliced()
        {
            var shape = new Shape(3, 3).Slice("1:");
            // 0 1 2
            //[[3 4 5]
            // [6 7 8]]
            shape.Should().BeShaped(2, 3);
            shape.GetOffset(0,0).Should().Be(3);
            shape.GetCoordinatesFromAbsoluteIndex(3).Should().Equal(new int[] { 0, 0 });
            shape.GetOffset(1, 2).Should().Be(8);
            shape.GetCoordinatesFromAbsoluteIndex(8).Should().Equal(new int[] { 1, 2 });
            shape = new Shape(3, 3).Slice(":, ::2");
            //[[0] 1 [2]
            // [3] 4 [5]
            // [6] 7 [8]]
            shape.Should().BeShaped(3, 2);
            shape.GetOffset(0, 1).Should().Be(2);
            shape.GetCoordinatesFromAbsoluteIndex(2).Should().Equal(new int[] { 0, 1 });
            shape.GetOffset(2, 1).Should().Be(8);
            shape.GetCoordinatesFromAbsoluteIndex(8).Should().Equal(new int[] { 2, 1 });
        }

        [Test]
        public void GetCoordinatesFromAbsoluteIndex_Sliced_by_Index()
        {
            var shape = new Shape(3, 3).Slice(Slice.Index(1));
            // 0 1 2
            //[3 4 5]
            // 6 7 8
            shape.Should().BeShaped(3);
            shape.GetOffset(1).Should().Be(4);
            shape.GetCoordinatesFromAbsoluteIndex(4).Should().Equal(new int[] { 1 });
        }

        private Shape ReshapeSlicedShape(Shape shape, params int[] new_dims)
        {
            Shape newShape = new Shape(new_dims);
            if (shape.IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() { ParentShape = shape, Slices = null };
            return newShape;
        } 

        [Test]
        public void GetCoordinatesFromAbsoluteIndex_Sliced_and_Reshaped()
        {
            //>>> a
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> a[:, 1:]
            //array([[1, 2],
            //       [4, 5],
            //       [7, 8]])
            //>>> a[:, 1:].reshape(2,3)
            //array([[1, 2, 4],
            //       [5, 7, 8]])
            var shape = ReshapeSlicedShape(new Shape(3, 3).Slice(":, 1:"), 2,3);
            shape.Should().BeShaped(2,3);
            //shape.GetOffset(0, 0).Should().Be(1);
            //shape.GetCoordinates(1).Should().Equal(new int[] { 0, 0 });
            shape.GetOffset(1,1).Should().Be(7);
            shape.GetCoordinatesFromAbsoluteIndex(7).Should().Equal(new int[] { 1, 1 });
            shape.GetOffset(1, 2).Should().Be(8);
            shape.GetCoordinatesFromAbsoluteIndex(8).Should().Equal(new int[] { 1, 2 });
            // now slice again:
            //>>> c
            //array([[1, 2, 4],
            //       [5, 7, 8]])
            //>>> c[1, 1:]
            //array([7, 8])
            var shape1 = shape.Slice("1, 1:");
            shape1.GetOffset(1).Should().Be(8);
            shape1.GetCoordinatesFromAbsoluteIndex(8).Should().Equal(new int[] { 1 });
            shape1.GetOffset(0).Should().Be(7);
            shape1.GetCoordinatesFromAbsoluteIndex(7).Should().Equal(new int[] { 0 });
        }

        #endregion

      
    }
}
