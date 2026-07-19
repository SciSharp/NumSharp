using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.block — assemble an nd-array from nested lists of blocks.
    ///     Every expected value below was probed against NumPy 2.4.2
    ///     (numpy/_core/shape_base.py::block).
    /// </summary>
    [TestClass]
    public class np_block_test : TestClass
    {
        // -- scalars as leaves (Python: np.block([[1, 2], [3, 4]]) ≡ np.array) --

        [TestMethod]
        public void Scalars_Depth1()
        {
            // np.block([1, 2, 3]) -> array([1, 2, 3])
            var r = np.block(new[] { 1, 2, 3 });
            r.shape.Should().Equal(3L);
            r.Data<int>().Should().Equal(1, 2, 3);
        }

        [TestMethod]
        public void Scalars_Depth2_Jagged()
        {
            // np.block([[1, 2], [3, 4]]) -> array([[1, 2], [3, 4]])
            var r = np.block(new[] { new[] { 1, 2 }, new[] { 3, 4 } });
            r.shape.Should().Equal(2L, 2L);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void Scalars_Depth2_ObjectArrays()
        {
            var r = np.block(new object[] { new object[] { 1, 2 }, new object[] { 3, 4 } });
            r.shape.Should().Equal(2L, 2L);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void ListOfT_ActsAsPythonList()
        {
            var r = np.block(new List<object> { new List<object> { 1, 2 }, new List<object> { 3, 4 } });
            r.shape.Should().Equal(2L, 2L);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void RectangularArray_IsALeaf()
        {
            // int[,] has no Python-list analog -> ndarray leaf (depth 0 -> copy)
            var r = np.block(new int[,] { { 1, 2 }, { 3, 4 } });
            r.shape.Should().Equal(2L, 2L);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        // -- hstack/vstack equivalents --

        [TestMethod]
        public void Depth1_ActsAsHStack_WithScalarMix()
        {
            // np.block([a, b, 10]) -> [1, 2, 3, 4, 5, 6, 10]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            var r = np.block(new object[] { a, b, 10 });
            r.Data<int>().Should().Equal(1, 2, 3, 4, 5, 6, 10);
        }

        [TestMethod]
        public void Depth2_ActsAsVStack()
        {
            // np.block([[a], [b]]) -> [[1, 2, 3], [4, 5, 6]]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            var r = np.block(new[] { new[] { a }, new[] { b } });
            r.shape.Should().Equal(2L, 3L);
            r.Data<int>().Should().Equal(1, 2, 3, 4, 5, 6);
        }

        // -- the canonical block-matrix example (NumPy docstring) --

        [TestMethod]
        public void BlockMatrix_Canonical()
        {
            var A = np.eye(2) * 2;
            var B = np.eye(3) * 3;
            var r = np.block(new[]
            {
                new[] { A, np.zeros((2, 3)) },
                new[] { np.ones((3, 2)), B }
            });
            r.shape.Should().Equal(5L, 5L);
            r.Data<double>().Should().Equal(
                2, 0, 0, 0, 0,
                0, 2, 0, 0, 0,
                1, 1, 3, 0, 0,
                1, 1, 0, 3, 0,
                1, 1, 0, 0, 3);
        }

        [TestMethod]
        public void LeadingOnes_PadLowerNdimBlocks()
        {
            // np.block([[A, zeros(2,1)], [v, 30]]) -> [[1,1,0],[1,1,0],[10,20,30]]
            var A = np.ones((2, 2), np.int32);
            var v = np.array(new[] { 10, 20 });
            var r = np.block(new object[]
            {
                new object[] { A, np.zeros((2, 1), np.int32) },
                new object[] { v, 30 }
            });
            r.shape.Should().Equal(3L, 3L);
            r.Data<int>().Should().Equal(1, 1, 0, 1, 1, 0, 10, 20, 30);
        }

        // -- atleast_1d / atleast_2d equivalents --

        [TestMethod]
        public void SingletonNesting_ActsAsAtLeastNd()
        {
            np.block(new object[] { np.asanyarray(0) }).shape.Should().Equal(1L);
            np.block(new object[] { new object[] { np.asanyarray(5) } }).shape.Should().Equal(1L, 1L);
            np.block(new object[] { 7 }).shape.Should().Equal(1L);
            np.block(new object[] { new object[] { new object[] { 7 } } }).shape.Should().Equal(1L, 1L, 1L);
        }

        // -- depth 0 (single array / scalar) --

        [TestMethod]
        public void Depth0_SingleArray_ReturnsCopy()
        {
            // Probed NumPy 2.4.2: np.block(x).base is None and writing the
            // result does NOT modify x (despite the docstring's "not copied").
            var x = np.arange(4);
            var r = np.block(x);
            ReferenceEquals(r, x).Should().BeFalse();
            r.SetData(99, 0);
            ((int)x.GetData(new int[] { 0 })).Should().Be(0, "np.block of a bare array returns a copy");
        }

        [TestMethod]
        public void Depth0_Scalar()
        {
            var r = np.block(7);
            r.ndim.Should().Be(0);
            ((int)r).Should().Be(7);
        }

        // -- deep nesting: concatenation axes -1, -2, -3 --

        [TestMethod]
        public void NestingDepth_ControlsConcatAxis_3D()
        {
            var c = np.arange(8).reshape(2, 2, 2);
            np.block(new[] { c, c }).shape.Should().Equal(2L, 2L, 4L);
            np.block(new[] { new[] { c }, new[] { c } }).shape.Should().Equal(2L, 4L, 2L);
            np.block(new[] { new[] { new[] { c } }, new[] { new[] { c } } }).shape.Should().Equal(4L, 2L, 2L);
        }

        [TestMethod]
        public void NestingDeeperThanNdim_PrependsOnes()
        {
            var A = np.ones((2, 2), np.int32);
            // depth-3 nesting of 2-D leaves -> result ndim 3: (2, 2, 2)
            np.block(new[] { new[] { new[] { A } }, new[] { new[] { A } } }).shape.Should().Equal(2L, 2L, 2L);
        }

        // -- empty arrays --

        [TestMethod]
        public void EmptyLeaves()
        {
            np.block(new object[] { np.zeros(0), np.zeros(0) }).shape.Should().Equal(0L);
            np.block(new[] { new[] { np.zeros((0, 3)) }, new[] { np.zeros((0, 3)) } }).shape.Should().Equal(0L, 3L);
            // 1-D empties pad to (1, 0) then stack vertically -> (2, 0)
            np.block(new[] { new[] { np.zeros(0) }, new[] { np.zeros(0) } }).shape.Should().Equal(2L, 0L);
            // (2,0) contributes nothing along the concat axis
            var r = np.block(new object[] { np.zeros((2, 0)), np.ones((2, 3)) });
            r.shape.Should().Equal(2L, 3L);
            r.Data<double>().Should().Equal(1, 1, 1, 1, 1, 1);
        }

        // -- dtype promotion (NEP50 result_type over the blocks) --

        [TestMethod]
        public void DtypePromotion()
        {
            np.block(new object[] { np.array(new byte[] { 1 }), np.array(new sbyte[] { 2 }) })
                .typecode.Should().Be(NPTypeCode.Int16);
            np.block(new[] { new[] { np.array(new byte[] { 1 }) }, new[] { np.array(new float[] { 2.5f }) } })
                .typecode.Should().Be(NPTypeCode.Single);
            np.block(new object[] { np.array(new long[] { 1 }), 2.5 })
                .typecode.Should().Be(NPTypeCode.Double);
            np.block(new object[] { true, np.array(new byte[] { 2 }) })
                .typecode.Should().Be(NPTypeCode.Byte);
        }

        [TestMethod]
        public void AllDtypes_RoundTrip()
        {
            // block [[x], [x]] per dtype: (2, 2) result preserving dtype
            foreach (NPTypeCode tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
                NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
                NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            })
            {
                var x = np.ones(new Shape(1, 2), tc);
                var r = np.block(new[] { new[] { x }, new[] { x } });
                r.typecode.Should().Be(tc, $"dtype {tc} should be preserved");
                r.shape.Should().Equal(2L, 2L);
            }
        }

        // -- memory-order selection --

        [TestMethod]
        public void FortranInputs_ProduceFortranOutput()
        {
            var r = np.block(new[]
            {
                new[] { np.asfortranarray(np.eye(2)) },
                new[] { np.asfortranarray(np.eye(2)) }
            });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void CInputs_ProduceCOutput()
        {
            np.block(new[] { new[] { np.eye(2) }, new[] { np.eye(2) } })
                .Shape.IsContiguous.Should().BeTrue();
        }

        // -- errors (messages verbatim from NumPy 2.4.2) --

        [TestMethod]
        public void MismatchedDepths_Throws()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            var act = () => np.block(new object[] { new object[] { a, b }, a });
            act.Should().Throw<ValueError>().WithMessage(
                "List depths are mismatched. First element was at depth 2, but there is an element at depth 1 (arrays[1])");
        }

        [TestMethod]
        public void EmptyList_Throws()
        {
            var act = () => np.block(new object[0]);
            act.Should().Throw<ValueError>().WithMessage("List at arrays cannot be empty");
        }

        [TestMethod]
        public void NestedEmptyList_Throws()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var act = () => np.block(new object[] { new object[] { a }, new object[0] });
            act.Should().Throw<ValueError>().WithMessage("List at arrays[1] cannot be empty");
        }

        [TestMethod]
        public void Tuple_Throws()
        {
            var a = np.array(new[] { 1, 2 });
            var b = np.array(new[] { 3, 4 });
            var act = () => np.block((a, b));
            act.Should().Throw<TypeError>().WithMessage(
                "arrays is a tuple. Only lists can be used to arrange blocks, and np.block does not allow implicit conversion from tuple to ndarray.");
        }

        [TestMethod]
        public void NestedTuple_Throws_WithIndexedName()
        {
            var a = np.array(new[] { 1, 2 });
            var b = np.array(new[] { 3, 4 });
            var act = () => np.block(new object[] { (a, b) });
            act.Should().Throw<TypeError>().WithMessage("arrays[0] is a tuple.*");
        }

        [TestMethod]
        public void ShapeMismatch_ThrowsConcatenateMessage()
        {
            var act = () => np.block(new[]
            {
                new[] { np.ones((2, 2), np.int32) },
                new[] { np.ones((3, 3), np.int32) }
            });
            act.Should().Throw<IncorrectShapeException>().WithMessage(
                "*along dimension 1, the array at index 0 has size 2 and the array at index 1 has size 3*");
        }

        // -- the _block_slicing path (list_ndim * final_size > 2*512*512) --

        [TestMethod]
        public void SlicingPath_LargeBlocks_MatchConcatenatePath()
        {
            // 600x800 doubles -> 2 * 480000 = 960000 > 524288 forces the
            // single-allocation slicing algorithm; the nested-concatenate
            // reference must agree bit-for-bit.
            var l1 = np.ones((400, 500));
            var l2 = np.ones((400, 300)) * 2.0;
            var l3 = np.ones((200, 500)) * 3.0;
            var l4 = np.ones((200, 300)) * 4.0;

            var viaBlock = np.block(new[] { new[] { l1, l2 }, new[] { l3, l4 } });
            var viaConcat = np.concatenate(new[]
            {
                np.concatenate(new[] { l1, l2 }, 1),
                np.concatenate(new[] { l3, l4 }, 1)
            }, 0);

            viaBlock.shape.Should().Equal(600L, 800L);
            ((bool)np.array_equal(viaBlock, viaConcat)).Should().BeTrue();
        }

        [TestMethod]
        public void SlicingPath_MixedDtype_Promotes()
        {
            var big = np.block(new[] { new[] { np.ones((600, 600), np.int32), np.ones((600, 600)) * 2.0 } });
            big.typecode.Should().Be(NPTypeCode.Double);
            big.shape.Should().Equal(600L, 1200L);
            ((double)big.GetData(new int[] { 0, 0 })).Should().Be(1.0);
            ((double)big.GetData(new int[] { 599, 1199 })).Should().Be(2.0);
        }

        [TestMethod]
        public void SlicingPath_FortranInputs_ProduceFortranOutput()
        {
            var f1 = np.asfortranarray(np.ones((600, 600)));
            var f2 = np.asfortranarray(np.ones((600, 600)) * 2.0);
            var r = np.block(new[] { new[] { f1, f2 } });
            r.Shape.IsFContiguous.Should().BeTrue();
            ((double)r.GetData(new int[] { 5, 700 })).Should().Be(2.0);
        }

        // -- differential case pinned from live NumPy 2.4.2 --

        [TestMethod]
        public void Differential_MixedDtypeBlockMatrix_MatchesNumPy()
        {
            // rng = np.random.RandomState(42); A=(rand(3,4)*100).astype(i4);
            // B=(rand(3,2)*100).astype(i4); C=(rand(2,4)*100).astype(f4);
            // D=(rand(2,2)*100).astype(f8); r = np.block([[A, B], [C, D]])
            // -> dtype f8, shape (5,6), sum 1307.7046511861654,
            //    r[0,0]=37.0, r[4,5]=4.645041271999773, r[3,3]=13.949385643005371
            np.random.seed(42);
            var A = (np.random.rand(3, 4) * 100).astype(np.int32);
            var B = (np.random.rand(3, 2) * 100).astype(np.int32);
            var C = (np.random.rand(2, 4) * 100).astype(np.float32);
            var D = (np.random.rand(2, 2) * 100).astype(np.float64);

            var r = np.block(new[] { new[] { A, B }, new[] { C, D } });
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().Equal(5L, 6L);
            ((double)np.sum(r)).Should().Be(1307.7046511861654);
            ((double)r.GetData(new int[] { 0, 0 })).Should().Be(37.0);
            ((double)r.GetData(new int[] { 4, 5 })).Should().Be(4.645041271999773);
            ((double)r.GetData(new int[] { 3, 3 })).Should().Be(13.949385643005371);
        }

        // -- views/strided leaves --

        [TestMethod]
        public void StridedAndTransposedLeaves()
        {
            var baseArr = np.arange(24).reshape(4, 6);
            var s = baseArr["::2"];        // (2, 6) strided view
            var t = np.arange(12).reshape(6, 2).T; // (2, 6) transposed view
            var r = np.block(new[] { new[] { s }, new[] { t } });
            r.shape.Should().Equal(4L, 6L);
            // row 0 = baseArr row 0; row 2 = t row 0 = [0, 2, 4, 6, 8, 10]
            ((int)r.GetData(new int[] { 0, 5 })).Should().Be(5);
            ((int)r.GetData(new int[] { 1, 0 })).Should().Be(12);
            ((int)r.GetData(new int[] { 2, 1 })).Should().Be(2);
            ((int)r.GetData(new int[] { 3, 5 })).Should().Be(11);
        }

        [TestMethod]
        public void Null_Throws()
        {
            var act = () => np.block(null);
            act.Should().Throw<ArgumentNullException>();
        }
    }
}
