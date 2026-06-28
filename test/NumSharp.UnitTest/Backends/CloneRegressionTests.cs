using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends
{
    [TestClass]
    public unsafe class CloneRegressionTests
    {
        private sealed class TestEngine : DefaultEngine { }

        [TestMethod]
        public void ArraySlice_CopyToSpan_CopiesFromSliceToDestination()
        {
            var source = ArraySlice.FromArray(new[] { 1, 2, 3 }, copy: true);
            var destination = new[] { -1, -1, -1 };

            source.CopyTo(destination.AsSpan());

            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, destination);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, source.ToArray());
        }

        [TestMethod]
        public void ArraySlice_TryCopyToSpan_CopiesFromSliceToDestination()
        {
            var source = ArraySlice.FromArray(new[] { 4, 5, 6 }, copy: true);
            var destination = new[] { -1, -1, -1 };

            Assert.IsTrue(source.TryCopyTo(destination.AsSpan()));

            CollectionAssert.AreEqual(new[] { 4, 5, 6 }, destination);
            CollectionAssert.AreEqual(new[] { 4, 5, 6 }, source.ToArray());
        }

        [TestMethod]
        public void ArraySlice_CopyToSpan_WithSourceRange_CopiesRequestedRange()
        {
            var source = ArraySlice.FromArray(new[] { 10, 20, 30, 40, 50 }, copy: true);
            var destination = new[] { -1, -1 };

            source.CopyTo(destination.AsSpan(), sourceOffset: 2, sourceLength: 2);

            CollectionAssert.AreEqual(new[] { 30, 40 }, destination);
            CollectionAssert.AreEqual(new[] { 10, 20, 30, 40, 50 }, source.ToArray());
        }

        [TestMethod]
        public void ArraySlice_CopyToIntPtr_WithSourceRange_CopiesRequestedRange()
        {
            var source = ArraySlice.FromArray(new[] { 10, 20, 30, 40, 50 }, copy: true);
            var destination = new[] { -1, -1, -1, -1, -1 };

            fixed (int* destinationPtr = destination)
            {
                source.CopyTo((IntPtr)destinationPtr, sourceOffset: 1, sourceCount: 3);
            }

            CollectionAssert.AreEqual(new[] { 20, 30, 40, -1, -1 }, destination);
            CollectionAssert.AreEqual(new[] { 10, 20, 30, 40, 50 }, source.ToArray());
        }

        [TestMethod]
        public void ArraySlice_InterfaceCopyToSpan_CopiesFromSliceToDestination()
        {
            IArraySlice source = ArraySlice.FromArray(new[] { 7, 8, 9 }, copy: true);
            var destination = new[] { -1, -1, -1 };

            source.CopyTo<int>(destination.AsSpan());

            CollectionAssert.AreEqual(new[] { 7, 8, 9 }, destination);
            CollectionAssert.AreEqual(new[] { 7, 8, 9 }, source.ToArray());
        }

        [TestMethod]
        public void ArraySlice_InterfaceCloneGeneric_ReinterpretsWholeBytePayload()
        {
            IArraySlice source = ArraySlice.FromArray(new[] { 0x11223344, 0x55667788 }, copy: true);

            var bytes = source.Clone<byte>();

            Assert.AreEqual(8, bytes.Count);
        }

        [TestMethod]
        public void ArrayConvert_Clone_PreservesJaggedElementType()
        {
            var source = new[] { new[] { 1, 2 }, new[] { 3 } };

            var clone = ArrayConvert.Clone(source);

            Assert.IsInstanceOfType(clone, typeof(int[][]));
            Assert.AreNotSame(source, clone);
            Assert.AreSame(source[0], ((int[][])clone)[0]);
            Assert.AreSame(source[1], ((int[][])clone)[1]);
        }

        [TestMethod]
        public void ArrayConvert_Clone_PreservesNonZeroLowerBounds()
        {
            var source = Array.CreateInstance(typeof(int), new[] { 3 }, new[] { 5 });
            source.SetValue(10, 5);
            source.SetValue(20, 6);
            source.SetValue(30, 7);

            var clone = ArrayConvert.Clone(source);

            Assert.AreEqual(5, clone.GetLowerBound(0));
            Assert.AreEqual(7, clone.GetUpperBound(0));
            Assert.AreEqual(20, clone.GetValue(6));
        }

        [TestMethod]
        public void ArrayConvert_Clone_FourDimensionalArray_UsesFourthDimensionLength()
        {
            var source = new int[1, 2, 3, 4];
            source[0, 1, 2, 3] = 42;

            var clone = ArrayConvert.Clone(source);

            Assert.AreEqual(4, clone.GetLength(3));
            Assert.AreEqual(42, clone[0, 1, 2, 3]);
        }

        [TestMethod]
        public void Shape_Clone_PreservesScalarViewOffset()
        {
            var scalar = np.arange(10)["5"];

            var clone = scalar.Shape.Clone();

            Assert.IsTrue(clone.IsScalar);
            Assert.AreEqual(scalar.Shape.offset, clone.offset);
            Assert.AreEqual(scalar.Shape.bufferSize, clone.bufferSize);
        }

        [TestMethod]
        public void UnmanagedStorage_Clone_DtypeOnlyStorage_DoesNotDereferenceMissingData()
        {
            var engine = new TestEngine();
            var storage = new UnmanagedStorage(NPTypeCode.Int32) { Engine = engine };

            var clone = storage.Clone();

            Assert.AreEqual(NPTypeCode.Int32, clone.TypeCode);
            Assert.AreSame(engine, clone.Engine);
            Assert.IsNull(clone.InternalArray);
        }

        [TestMethod]
        public void UnmanagedStorage_Clone_PreservesEngineAndFContiguousShape()
        {
            var engine = new TestEngine();
            var storage = new UnmanagedStorage(NPTypeCode.Int32) { Engine = engine };
            storage.Allocate(new Shape(new long[] { 3, 4 }, 'F'), NPTypeCode.Int32, fillZeros: true);

            var clone = storage.Clone();

            Assert.AreSame(engine, clone.Engine);
            Assert.IsTrue(clone.Shape.IsFContiguous);
            Assert.IsFalse(clone.Shape.IsContiguous);
        }

        [TestMethod]
        public void UnmanagedStorage_CastTypeCode_FContiguousSource_CopiesLogicalValuesAndEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(6).reshape(2, 3).T;
            source.TensorEngine = engine;

            var cast = new NDArray(source.Storage.Cast(NPTypeCode.Double));

            Assert.AreSame(engine, cast.TensorEngine);
            Assert.AreSame(engine, cast.Storage.Engine);
            Assert.AreEqual(0.0, (double)cast[0, 0]);
            Assert.AreEqual(3.0, (double)cast[0, 1]);
            Assert.AreEqual(1.0, (double)cast[1, 0]);
            Assert.AreEqual(5.0, (double)cast[2, 1]);
        }

        [TestMethod]
        public void UnmanagedStorage_CastGeneric_FContiguousSource_CopiesLogicalValuesAndEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(6).reshape(2, 3).T;
            source.TensorEngine = engine;

            var cast = new NDArray(source.Storage.Cast<double>());

            Assert.AreSame(engine, cast.TensorEngine);
            Assert.AreSame(engine, cast.Storage.Engine);
            Assert.AreEqual(0.0, (double)cast[0, 0]);
            Assert.AreEqual(3.0, (double)cast[0, 1]);
            Assert.AreEqual(1.0, (double)cast[1, 0]);
            Assert.AreEqual(5.0, (double)cast[2, 1]);
        }

        [TestMethod]
        public void UnmanagedStorage_CastIfNecessary_FContiguousSource_CopiesLogicalValuesAndEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(6).reshape(2, 3).T;
            source.TensorEngine = engine;

            var cast = new NDArray(source.Storage.CastIfNecessary(NPTypeCode.Double));

            Assert.AreSame(engine, cast.TensorEngine);
            Assert.AreSame(engine, cast.Storage.Engine);
            Assert.AreEqual(0.0, (double)cast[0, 0]);
            Assert.AreEqual(3.0, (double)cast[0, 1]);
            Assert.AreEqual(1.0, (double)cast[1, 0]);
            Assert.AreEqual(5.0, (double)cast[2, 1]);
        }

        [TestMethod]
        public void UnmanagedStorage_CastEmptyStorage_PreservesEngine()
        {
            var engine = new TestEngine();
            var storage = new UnmanagedStorage(NPTypeCode.Int32) { Engine = engine };

            var cast = storage.Cast(NPTypeCode.Double);
            var genericCast = storage.Cast<double>();
            var castIfNecessary = storage.CastIfNecessary(NPTypeCode.Double);

            Assert.AreSame(engine, cast.Engine);
            Assert.AreSame(engine, genericCast.Engine);
            Assert.AreSame(engine, castIfNecessary.Engine);
            Assert.AreEqual(NPTypeCode.Double, castIfNecessary.TypeCode);
        }

        [TestMethod]
        public void UnmanagedMemoryBlock_CopyToWithIndex_CopiesToDestinationOffset()
        {
            var source = UnmanagedMemoryBlock<int>.FromArray(new[] { 1, 2 });
            var destination = UnmanagedMemoryBlock<int>.FromArray(new[] { 9, 9, 9, 9 });

            source.CopyTo(destination, arrayIndex: 1);

            var actual = new int[4];
            destination.CopyTo(actual, 0);
            CollectionAssert.AreEqual(new[] { 9, 1, 2, 9 }, actual);
        }

        [TestMethod]
        public void UnmanagedHelper_CopyToWithInvalidDestinationOffset_Throws()
        {
            var source = new UnmanagedMemoryBlock<int>(0);
            var destination = UnmanagedMemoryBlock<int>.FromArray(new[] { 9, 9 });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
                UnmanagedHelper.CopyTo((IMemoryBlock)source, (IMemoryBlock)destination, countOffsetDestination: 3));
        }

        [TestMethod]
        public void NDArray_Clone_PreservesGenericRuntimeType()
        {
            NDArray source = np.array(new[] { 1, 2, 3 }).MakeGeneric<int>();

            var clone = source.Clone();

            Assert.IsInstanceOfType(clone, typeof(NDArray<int>));
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, clone.ToArray<int>());
        }

        [TestMethod]
        public void NDArray_Clone_PreservesTensorEngineOnArrayAndStorage()
        {
            var engine = new TestEngine();
            var source = np.arange(3);
            source.TensorEngine = engine;

            var clone = source.Clone();

            Assert.AreSame(engine, clone.TensorEngine);
            Assert.AreSame(engine, clone.Storage.Engine);
        }

        [TestMethod]
        public void NpArray_FromNDArray_PreservesTensorEngineForAliasAndCopy()
        {
            var engine = new TestEngine();
            var source = np.arange(3);
            source.TensorEngine = engine;

            var alias = np.array(source, copy: false);
            var copy = np.array(source, copy: true);

            Assert.AreSame(engine, alias.TensorEngine);
            Assert.AreSame(engine, alias.Storage.Engine);
            Assert.AreSame(engine, copy.TensorEngine);
            Assert.AreSame(engine, copy.Storage.Engine);
        }

        [TestMethod]
        public void NDArray_CopyFOrder_PreservesTensorEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(12).reshape(3, 4);
            source.TensorEngine = engine;

            var copy = source.copy('F');

            Assert.IsTrue(copy.Shape.IsFContiguous);
            Assert.AreSame(engine, copy.TensorEngine);
            Assert.AreSame(engine, copy.Storage.Engine);
        }

        [TestMethod]
        public void NDArray_CopyCOrder_FromFContiguousSource_ProducesCContiguousCopy()
        {
            var source = np.arange(12).reshape(3, 4).T;

            var copy = source.copy('C');

            Assert.IsTrue(copy.Shape.IsContiguous);
            Assert.IsFalse(copy.Shape.IsFContiguous && !copy.Shape.IsContiguous);
            Assert.AreEqual(0, (int)copy[0, 0]);
            Assert.AreEqual(11, (int)copy[3, 2]);
        }

        [TestMethod]
        public void NDArray_ReshapeCopyPath_PreservesTensorEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(12).reshape(3, 4).T;
            source.TensorEngine = engine;

            var reshaped = source.reshape(12);

            Assert.AreSame(engine, reshaped.TensorEngine);
            Assert.AreSame(engine, reshaped.Storage.Engine);
        }

        [TestMethod]
        public void NDArray_AstypeCopyPath_PreservesTensorEngine()
        {
            var engine = new TestEngine();
            var source = np.arange(3);
            source.TensorEngine = engine;

            var cast = source.astype(NPTypeCode.Double, copy: true);

            Assert.AreSame(engine, cast.TensorEngine);
            Assert.AreSame(engine, cast.Storage.Engine);
        }

        [TestMethod]
        public void NDIterCopy_BufferedIterator_AllocatesIndependentBuffers()
        {
            // NOTE: a same-dtype linear-strided operand is no longer buffered
            // (NumPy parity: 'buffered' enables buffering only when REQUIRED —
            // cast / non-linear layout). Request float64 op_dtypes over the
            // int32 source so the operand genuinely needs a cast buffer,
            // preserving this test's purpose: Copy() must duplicate the
            // buffer storage, not alias it.
            var source = np.arange(16)["::2"].astype(np.int32);

            using var iter = NDIterRef.AdvancedNew(
                nop: 1,
                op: new[] { source },
                flags: NDIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NDIterPerOpFlags.READONLY },
                opDtypes: new[] { NPTypeCode.Double },
                bufferSize: 4);

            using var copy = iter.Copy();

            Assert.AreNotEqual((nint)iter.GetDataPtr(0), (nint)copy.GetDataPtr(0));
            Assert.AreEqual(iter.GetValue<double>(0), copy.GetValue<double>(0));

            Assert.IsTrue(iter.Iternext());
            Assert.AreEqual(1, iter.IterIndex);
            Assert.AreEqual(0, copy.IterIndex);
        }

        [TestMethod]
        public void NDIterCopy_AfterRemoveAxis_PreservesAllocatedStrideWidth()
        {
            var source = np.arange(24).reshape(2, 3, 4);

            using var iter = NDIterRef.New(source, NDIterGlobalFlags.MULTI_INDEX);
            Assert.IsTrue(iter.RemoveAxis(1));

            using var copy = iter.Copy();

            Assert.AreEqual(iter.NDim, copy.NDim);
            CollectionAssert.AreEqual(iter.Shape, copy.Shape);
            Assert.AreEqual(iter.GetValue<int>(0), copy.GetValue<int>(0));
        }
    }
}
