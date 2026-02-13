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

        // Test removed: ChangeTensorLayout was removed (NumSharp is C-order only)
        // [Test, Skip("Ignored")]
        // public void CheckColRowSwitch() { ... }

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
            // NumPy-pure: test offset and strides instead of ViewInfo
            new Shape(10).Slice(":").Should().Be(new Shape(10));
            new Shape(10).Slice("-77:77").Should().Be(new Shape(10));
            new Shape(10).Slice(":7").Dimensions[0].Should().Be(7);
            new Shape(10).Slice("7:").Dimensions[0].Should().Be(3);
            new Shape(10).Slice("7:").Offset.Should().Be(7);
            new Shape(10).Slice("-7:").Dimensions[0].Should().Be(7);
            new Shape(10).Slice("-7:").Offset.Should().Be(3);
        }

        [Test]
        public void RepeatedSlicing_1D()
        {
            // NumPy-pure: double slicing uses parent offset+strides
            new Shape(10).Slice(":").Slice(":").Dimensions[0].Should().Be(10);
            new Shape(10).Slice(":5").Slice("2:").Dimensions[0].Should().Be(3);
            new Shape(10).Slice(":5").Slice("2:").Offset.Should().Be(2);
            new Shape(10).Slice(":5").Slice("2:").Slice("::2").Dimensions[0].Should().Be(2);
            new Shape(10).Slice(":5").Slice("2:").Slice("::2").Strides[0].Should().Be(2);
            // bufferSize should track original buffer
            new Shape(10).Slice(":5").Slice("2:").BufferSize.Should().Be(10);
        }

        [Test]
        public void ShapeSlicing_2D()
        {
            // NumPy-pure: test dimensions and offset
            new Shape(3, 3).Slice(":,1:").Should().Be(new Shape(3, 2));
            new Shape(3, 3).Slice(":,1:").Offset.Should().Be(1);
            new Shape(3, 3).Slice(":,1:").Strides[0].Should().Be(3);
            new Shape(3, 3).Slice(":,1:").Strides[1].Should().Be(1);
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
            // With readonly struct, hashcode is computed at construction time
            var a = Shape.Vector(5);
            var b = new Shape(5);
            a._hashCode.Should().Be(b._hashCode);

            a = Shape.Matrix(5, 10);
            b = new Shape(5, 10);
            a._hashCode.Should().Be(b._hashCode);

            a = new Shape(3, 3, 3);
            b = new Shape(3, 3, 3);
            a._hashCode.Should().Be(b._hashCode);
        }

        [Test]
        public void HashcodeScalars()
        {
            Shape.Scalar.GetHashCode().Should().Be(int.MinValue);
            Shape.NewScalar().GetHashCode().Should().Be(int.MinValue);
            // NumPy-pure: scalars with offset still have scalar hashcode
            // Use constructor to create scalar with offset (readonly struct)
            var scalarWithOffset = new Shape(Array.Empty<int>(), Array.Empty<int>(), 5, 10);
            scalarWithOffset.GetHashCode().Should().Be(int.MinValue);
        }

        // GetCoordinatesFromAbsoluteIndex region removed - method was dead API (not supported in NumPy for views)

      
    }
}
