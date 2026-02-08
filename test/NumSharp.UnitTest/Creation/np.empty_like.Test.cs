using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Comprehensive tests for np.empty_like, verified against NumPy 2.4.2 ground truth.
    ///
    /// NumPy signature: empty_like(prototype, dtype=None, order='K', subok=True, shape=None, *, device=None)
    /// NumSharp signatures:
    ///   empty_like(NDArray prototype, Type dtype = null, Shape shape = default)
    ///   empty_like(NDArray prototype, NPTypeCode typeCode, Shape shape = default)
    ///
    /// Key behaviors verified:
    /// - Shape and dtype are always preserved from prototype unless explicitly overridden
    /// - Result is always new memory (never shares with prototype)
    /// - Result is always writeable (even if prototype was read-only broadcast)
    /// - Content is uninitialized (no zeroing guarantee)
    /// - Since NumSharp is C-order only, result is always C-contiguous
    /// </summary>
    [TestClass]
    public class np_empty_like_Test
    {
        // ==========================================================
        // 1. BASIC: shape and dtype preservation
        // ==========================================================

        [TestMethod]
        public void Basic_1D_Int32_PreservesShapeAndDtype()
        {
            // NumPy: np.empty_like(np.arange(5, dtype='int32')) → shape=(5,), dtype=int32
            var a = np.arange(5).astype(np.int32);
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.dtype.Should().Be(typeof(int));
            r.ndim.Should().Be(1);
        }

        [TestMethod]
        public void Basic_2D_Float64_PreservesShapeAndDtype()
        {
            // NumPy: np.empty_like(np.arange(6, dtype='float64').reshape(2,3)) → shape=(2,3), dtype=float64
            var a = np.arange(6).astype(np.float64).reshape(2, 3);
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(double));
            r.ndim.Should().Be(2);
        }

        [TestMethod]
        public void Basic_3D_PreservesShapeAndDtype()
        {
            // NumPy: np.empty_like(np.arange(24).reshape(2,3,4)) → shape=(2,3,4)
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
            r.ndim.Should().Be(3);
        }

        [TestMethod]
        public void Basic_4D_PreservesShapeAndDtype()
        {
            // NumPy: np.empty_like(np.arange(120).reshape(2,3,4,5)) → shape=(2,3,4,5)
            var a = np.arange(120).reshape(2, 3, 4, 5);
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3, 4, 5 });
            r.ndim.Should().Be(4);
        }

        [TestMethod]
        public void Basic_Scalar_PreservesScalarShape()
        {
            // NumPy: np.empty_like(np.array(3.14)) → shape=(), dtype=float64, ndim=0
            // Construct scalar explicitly (np.array(3.14) may produce (1,) in NumSharp)
            var a = NDArray.Scalar(3.14);
            var r = np.empty_like(a);
            r.Shape.IsScalar.Should().BeTrue("result of empty_like on scalar should be scalar");
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Basic_SingleElement_PreservesShape()
        {
            // NumPy: np.empty_like(np.array([42], dtype='int32')) → shape=(1,), dtype=int32
            var a = np.array(new int[] { 42 });
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 1 });
            r.dtype.Should().Be(typeof(int));
        }

        // ==========================================================
        // 2. DTYPE OVERRIDE (Type overload)
        // ==========================================================

        [TestMethod]
        public void DtypeOverride_Int32ToFloat32()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(float));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 }, "shape preserved");
            r.dtype.Should().Be(typeof(float), "dtype overridden");
        }

        [TestMethod]
        public void DtypeOverride_Int32ToFloat64()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(double));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void DtypeOverride_Int32ToInt64()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(long));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void DtypeOverride_Int32ToInt16()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(short));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(short));
        }

        [TestMethod]
        public void DtypeOverride_Int32ToByte()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(byte));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(byte));
        }

        [TestMethod]
        public void DtypeOverride_Int32ToBool()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(bool));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(bool));
        }

        [TestMethod]
        public void DtypeOverride_Float64ToDecimal()
        {
            var a = np.arange(4).astype(np.float64).reshape(2, 2);
            var r = np.empty_like(a, typeof(decimal));
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            r.dtype.Should().Be(typeof(decimal));
        }

        [TestMethod]
        public void DtypeOverride_NullDtype_PreservesOriginal()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, (Type)null);
            r.dtype.Should().Be(typeof(int), "null dtype should preserve prototype's dtype");
        }

        // ==========================================================
        // 3. NPTypeCode OVERLOAD
        // ==========================================================

        [TestMethod]
        public void NPTypeCode_Overload_Float32()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Single);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void NPTypeCode_Overload_Float64()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Double);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void NPTypeCode_Overload_Boolean()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Boolean);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(bool));
        }

        [TestMethod]
        public void NPTypeCode_Overload_Int64()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Int64);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void NPTypeCode_Overload_Byte()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Byte);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.dtype.Should().Be(typeof(byte));
        }

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean, typeof(bool), DisplayName = "NPTypeCode_Boolean")]
        [DataRow(NPTypeCode.Byte, typeof(byte), DisplayName = "NPTypeCode_Byte")]
        [DataRow(NPTypeCode.Int16, typeof(short), DisplayName = "NPTypeCode_Int16")]
        [DataRow(NPTypeCode.UInt16, typeof(ushort), DisplayName = "NPTypeCode_UInt16")]
        [DataRow(NPTypeCode.Int32, typeof(int), DisplayName = "NPTypeCode_Int32")]
        [DataRow(NPTypeCode.UInt32, typeof(uint), DisplayName = "NPTypeCode_UInt32")]
        [DataRow(NPTypeCode.Int64, typeof(long), DisplayName = "NPTypeCode_Int64")]
        [DataRow(NPTypeCode.UInt64, typeof(ulong), DisplayName = "NPTypeCode_UInt64")]
        [DataRow(NPTypeCode.Char, typeof(char), DisplayName = "NPTypeCode_Char")]
        [DataRow(NPTypeCode.Single, typeof(float), DisplayName = "NPTypeCode_Single")]
        [DataRow(NPTypeCode.Double, typeof(double), DisplayName = "NPTypeCode_Double")]
        [DataRow(NPTypeCode.Decimal, typeof(decimal), DisplayName = "NPTypeCode_Decimal")]
        public void NPTypeCode_Overload_All12Types(NPTypeCode typeCode, Type expectedType)
        {
            var a = np.arange(4).reshape(2, 2);
            var r = np.empty_like(a, typeCode);
            r.dtype.Should().Be(expectedType);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
        }

        [TestMethod]
        public void NPTypeCode_Overload_WithShapeOverride()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, NPTypeCode.Double, new Shape(4, 5));
            r.shape.Should().BeEquivalentTo(new[] { 4, 5 });
            r.dtype.Should().Be(typeof(double));
        }

        // ==========================================================
        // 4. ALL 12 DTYPES preserved (Type overload, no dtype override)
        // ==========================================================

        [DataTestMethod]
        [DataRow(typeof(bool), DisplayName = "Boolean")]
        [DataRow(typeof(byte), DisplayName = "Byte")]
        [DataRow(typeof(short), DisplayName = "Int16")]
        [DataRow(typeof(ushort), DisplayName = "UInt16")]
        [DataRow(typeof(int), DisplayName = "Int32")]
        [DataRow(typeof(uint), DisplayName = "UInt32")]
        [DataRow(typeof(long), DisplayName = "Int64")]
        [DataRow(typeof(ulong), DisplayName = "UInt64")]
        [DataRow(typeof(char), DisplayName = "Char")]
        [DataRow(typeof(float), DisplayName = "Single")]
        [DataRow(typeof(double), DisplayName = "Double")]
        [DataRow(typeof(decimal), DisplayName = "Decimal")]
        public void AllDtypes_Preserved(Type dtype)
        {
            var a = np.ones(new Shape(3), dtype);
            var r = np.empty_like(a);
            r.dtype.Should().Be(dtype, $"dtype {dtype.Name} should be preserved");
            r.shape.Should().BeEquivalentTo(new[] { 3 });
        }

        [DataTestMethod]
        [DataRow(typeof(bool), DisplayName = "Boolean_2D")]
        [DataRow(typeof(byte), DisplayName = "Byte_2D")]
        [DataRow(typeof(short), DisplayName = "Int16_2D")]
        [DataRow(typeof(ushort), DisplayName = "UInt16_2D")]
        [DataRow(typeof(int), DisplayName = "Int32_2D")]
        [DataRow(typeof(uint), DisplayName = "UInt32_2D")]
        [DataRow(typeof(long), DisplayName = "Int64_2D")]
        [DataRow(typeof(ulong), DisplayName = "UInt64_2D")]
        [DataRow(typeof(char), DisplayName = "Char_2D")]
        [DataRow(typeof(float), DisplayName = "Single_2D")]
        [DataRow(typeof(double), DisplayName = "Double_2D")]
        [DataRow(typeof(decimal), DisplayName = "Decimal_2D")]
        public void AllDtypes_2D_Preserved(Type dtype)
        {
            var a = np.ones(new Shape(2, 3), dtype);
            var r = np.empty_like(a);
            r.dtype.Should().Be(dtype);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
        }

        // ==========================================================
        // 5. EMPTY ARRAYS (zero-size dimensions)
        // ==========================================================

        [TestMethod]
        public void Empty_1D_ZeroElements()
        {
            // NumPy: np.empty_like(np.array([], dtype='float64')) → shape=(0,), size=0
            var a = np.empty(new Shape(0), typeof(double));
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 0 });
            r.dtype.Should().Be(typeof(double));
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Empty_2D_ZeroRows()
        {
            // NumPy: np.empty_like(np.empty((0,3), dtype='int32')) → shape=(0,3), size=0
            var a = np.empty(new Shape(0, 3), typeof(int));
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 0, 3 });
            r.dtype.Should().Be(typeof(int));
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Empty_2D_ZeroCols()
        {
            // NumPy: np.empty_like(np.empty((3,0), dtype='int32')) → shape=(3,0), size=0
            var a = np.empty(new Shape(3, 0), typeof(int));
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 3, 0 });
            r.dtype.Should().Be(typeof(int));
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Empty_3D_ZeroMiddleDim()
        {
            // NumPy: np.empty_like(np.empty((2,0,4), dtype='float32')) → shape=(2,0,4), size=0
            var a = np.empty(new Shape(2, 0, 4), typeof(float));
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 2, 0, 4 });
            r.dtype.Should().Be(typeof(float));
            r.size.Should().Be(0);
        }

        // ==========================================================
        // 6. SLICED PROTOTYPES
        // ==========================================================

        [TestMethod]
        public void Sliced_1D_Contiguous()
        {
            // NumPy: np.empty_like(np.arange(10)[2:7]) → shape=(5,), C-contiguous
            var a = np.arange(10);
            var s = a["2:7"];
            s.shape.Should().BeEquivalentTo(new[] { 5 }, "sanity: slice shape");

            var r = np.empty_like(s);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.dtype.Should().Be(a.dtype);
            r.ndim.Should().Be(1);
        }

        [TestMethod]
        public void Sliced_1D_Stepped()
        {
            // NumPy: np.empty_like(np.arange(10)[::2]) → shape=(5,), C-contiguous
            var a = np.arange(10);
            var s = a["::2"];
            s.shape.Should().BeEquivalentTo(new[] { 5 }, "sanity: stepped slice shape");

            var r = np.empty_like(s);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
            r.dtype.Should().Be(a.dtype);
        }

        [TestMethod]
        public void Sliced_2D_RowSlice()
        {
            // NumPy: np.empty_like(np.arange(12).reshape(3,4)[1:3]) → shape=(2,4)
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 2, 4 }, "sanity: row slice shape");

            var r = np.empty_like(s);
            r.shape.Should().BeEquivalentTo(new[] { 2, 4 });
            r.dtype.Should().Be(a.dtype);
        }

        [TestMethod]
        public void Sliced_2D_ColumnSlice()
        {
            // NumPy: np.empty_like(np.arange(12).reshape(3,4)[:,1:3]) → shape=(3,2), C-contiguous
            var a = np.arange(12).reshape(3, 4);
            var s = a[":, 1:3"];
            s.shape.Should().BeEquivalentTo(new[] { 3, 2 }, "sanity: column slice shape");

            var r = np.empty_like(s);
            r.shape.Should().BeEquivalentTo(new[] { 3, 2 });
            r.dtype.Should().Be(a.dtype);
        }

        [TestMethod]
        public void Sliced_ReversedSlice()
        {
            // NumPy: np.empty_like(np.arange(5)[::-1]) → shape=(5,)
            var a = np.arange(5);
            var s = a["::-1"];
            s.shape.Should().BeEquivalentTo(new[] { 5 }, "sanity: reversed slice shape");

            var r = np.empty_like(s);
            r.shape.Should().BeEquivalentTo(new[] { 5 });
        }

        // ==========================================================
        // 7. BROADCAST PROTOTYPES
        // ==========================================================

        [TestMethod]
        public void Broadcast_RowVector()
        {
            // NumPy: np.empty_like(np.broadcast_to([1,2,3], (4,3))) → shape=(4,3), writeable
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(4, 3));
            b.shape.Should().BeEquivalentTo(new[] { 4, 3 }, "sanity: broadcast shape");

            var r = np.empty_like(b);
            r.shape.Should().BeEquivalentTo(new[] { 4, 3 });
            r.dtype.Should().Be(typeof(int));
            r.size.Should().Be(12);
        }

        [TestMethod]
        public void Broadcast_ScalarToMatrix()
        {
            // NumPy: np.empty_like(np.broadcast_to(5, (3,4))) → shape=(3,4)
            var a = np.array(5);
            var b = np.broadcast_to(a, new Shape(3, 4));
            b.shape.Should().BeEquivalentTo(new[] { 3, 4 }, "sanity: broadcast shape");

            var r = np.empty_like(b);
            r.shape.Should().BeEquivalentTo(new[] { 3, 4 });
            r.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Broadcast_ColumnVector()
        {
            // np.broadcast_to(np.array([[10],[20],[30]]), (3,3))
            var a = np.array(new int[] { 10, 20, 30 }).reshape(3, 1);
            var b = np.broadcast_to(a, new Shape(3, 3));
            b.shape.Should().BeEquivalentTo(new[] { 3, 3 }, "sanity: broadcast shape");

            var r = np.empty_like(b);
            r.shape.Should().BeEquivalentTo(new[] { 3, 3 });
            r.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Broadcast_WithDtypeOverride()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(4, 3));

            var r = np.empty_like(b, typeof(double));
            r.shape.Should().BeEquivalentTo(new[] { 4, 3 });
            r.dtype.Should().Be(typeof(double));
        }

        // ==========================================================
        // 8. TRANSPOSED PROTOTYPES
        // ==========================================================

        [TestMethod]
        public void Transposed_2D()
        {
            // NumPy: np.empty_like(np.arange(6).reshape(2,3).T) → shape=(3,2)
            var a = np.arange(6).reshape(2, 3);
            var t = a.T;
            t.shape.Should().BeEquivalentTo(new[] { 3, 2 }, "sanity: transposed shape");

            var r = np.empty_like(t);
            r.shape.Should().BeEquivalentTo(new[] { 3, 2 });
            r.dtype.Should().Be(a.dtype);
        }

        // ==========================================================
        // 9. MEMORY INDEPENDENCE — result never shares memory
        // ==========================================================

        [TestMethod]
        public void MemoryIndependence_PlainArray()
        {
            // NumPy: np.shares_memory(a, np.empty_like(a)) → False
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var r = np.empty_like(a);
            r.SetAtIndex(999, 0);
            a.GetAtIndex<int>(0).Should().Be(1, "writing to result must not affect prototype");
        }

        [TestMethod]
        public void MemoryIndependence_SlicedPrototype()
        {
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            var r = np.empty_like(s);
            r.SetAtIndex(999, 0);
            s.GetInt32(0, 0).Should().Be(4, "writing to result must not affect sliced prototype");
            a.GetInt32(1, 0).Should().Be(4, "writing to result must not affect original array");
        }

        [TestMethod]
        public void MemoryIndependence_BroadcastPrototype()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(4, 3));
            var r = np.empty_like(b);
            r.SetAtIndex(999, 0);
            b.GetInt32(0, 0).Should().Be(1, "writing to result must not affect broadcast prototype");
            a.GetAtIndex<int>(0).Should().Be(1, "writing to result must not affect original");
        }

        // ==========================================================
        // 10. WRITEABILITY — result is always writeable
        // ==========================================================

        [TestMethod]
        public void Writeable_FromBroadcastPrototype()
        {
            // Broadcast arrays in NumPy are read-only, but empty_like result is writeable
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(4, 3));

            var r = np.empty_like(b);
            // This should not throw — result must be writeable
            var writeAction = () => r.SetAtIndex(42, 0);
            writeAction.Should().NotThrow("empty_like result should be writeable");
            r.GetAtIndex<int>(0).Should().Be(42);
        }

        [TestMethod]
        public void Writeable_FromSlicedPrototype()
        {
            var a = np.arange(10);
            var s = a["::2"]; // stepped slice
            var r = np.empty_like(s);
            var writeAction = () => r.SetAtIndex(42, 0);
            writeAction.Should().NotThrow();
        }

        // ==========================================================
        // 11. SIZE CORRECTNESS
        // ==========================================================

        [TestMethod]
        public void Size_Matches_Prototype()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.empty_like(a);
            r.size.Should().Be(24);
        }

        [TestMethod]
        public void Size_WithDtypeOverride_MatchesPrototypeShape()
        {
            // dtype override changes element size but not count
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(double));
            r.size.Should().Be(6, "size should match prototype element count regardless of dtype");
        }

        // ==========================================================
        // 12. COMPARISON WITH ZEROS_LIKE AND ONES_LIKE
        //     (same shape/dtype contract, different initialization)
        // ==========================================================

        [TestMethod]
        public void SameContract_AsZerosLike()
        {
            var a = np.arange(6).astype(np.float64).reshape(2, 3);

            var e = np.empty_like(a);
            var z = np.zeros_like(a);

            e.shape.Should().BeEquivalentTo(z.shape, "empty_like and zeros_like should have same shape");
            e.dtype.Should().Be(z.dtype, "empty_like and zeros_like should have same dtype");
        }

        [TestMethod]
        public void SameContract_AsOnesLike()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);

            var e = np.empty_like(a);
            var o = np.ones_like(a);

            e.shape.Should().BeEquivalentTo(o.shape, "empty_like and ones_like should have same shape");
            e.dtype.Should().Be(o.dtype, "empty_like and ones_like should have same dtype");
        }

        // ==========================================================
        // 13. LARGE ARRAYS
        // ==========================================================

        [TestMethod]
        public void LargeArray_1000x1000()
        {
            var a = np.empty(new Shape(1000, 1000), typeof(double));
            var r = np.empty_like(a);
            r.shape.Should().BeEquivalentTo(new[] { 1000, 1000 });
            r.dtype.Should().Be(typeof(double));
            r.size.Should().Be(1_000_000);
        }

        // ==========================================================
        // 14. SCALAR DTYPES (ndim=0) — all supported types
        // ==========================================================

        [DataTestMethod]
        [DataRow(typeof(int), DisplayName = "Scalar_Int32")]
        [DataRow(typeof(double), DisplayName = "Scalar_Double")]
        [DataRow(typeof(bool), DisplayName = "Scalar_Boolean")]
        [DataRow(typeof(float), DisplayName = "Scalar_Single")]
        [DataRow(typeof(long), DisplayName = "Scalar_Int64")]
        [DataRow(typeof(byte), DisplayName = "Scalar_Byte")]
        public void Scalar_AllDtypes(Type dtype)
        {
            // NumPy: np.empty_like(np.array(42, dtype=...)) → shape=(), ndim=0
            var a = new NDArray(dtype, Shape.Scalar);
            var r = np.empty_like(a);
            r.Shape.IsScalar.Should().BeTrue();
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(dtype);
        }

        // ==========================================================
        // 15. RESULT IS NOT A VIEW — always a fresh allocation
        // ==========================================================

        [TestMethod]
        public void NotAView_PlainArray()
        {
            var a = np.arange(5).astype(np.int32);
            var r = np.empty_like(a);

            // Fill result and verify prototype is untouched
            for (int i = 0; i < 5; i++)
                r.SetAtIndex(100 + i, i);

            for (int i = 0; i < 5; i++)
                a.GetAtIndex<int>(i).Should().Be(i, $"prototype[{i}] should be unchanged");
        }

        [TestMethod]
        public void NotAView_SlicedArray()
        {
            var a = np.arange(10).astype(np.int32);
            var s = a["3:7"];
            var r = np.empty_like(s);

            for (int i = 0; i < 4; i++)
                r.SetAtIndex(100 + i, i);

            // Verify both original and slice untouched
            a.GetAtIndex<int>(3).Should().Be(3);
            a.GetAtIndex<int>(4).Should().Be(4);
            s.GetAtIndex<int>(0).Should().Be(3);
        }

        // ==========================================================
        // 16. CHAINED OPERATIONS — empty_like on empty_like result
        // ==========================================================

        [TestMethod]
        public void Chained_EmptyLikeOfEmptyLike()
        {
            var a = np.arange(6).astype(np.float64).reshape(2, 3);
            var r1 = np.empty_like(a);
            var r2 = np.empty_like(r1);

            r2.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r2.dtype.Should().Be(typeof(double));
        }

        // ==========================================================
        // 17. INTEGRATION — empty_like used in np.roll pattern
        // ==========================================================

        [TestMethod]
        public void Integration_RollPattern()
        {
            // np.roll uses empty_like internally for its result buffer
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.empty_like(a);

            result.shape.Should().BeEquivalentTo(a.shape);
            result.dtype.Should().Be(a.dtype);
            result.size.Should().Be(a.size);
        }

        [TestMethod]
        public void Integration_RollPattern_2D()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var result = np.empty_like(a);

            result.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            result.dtype.Should().Be(typeof(int));
            result.size.Should().Be(6);
        }

        // ==========================================================
        // 18. SHAPE OVERRIDE PARAMETER
        // ==========================================================

        [TestMethod]
        public void ShapeOverride_2DTo2D()
        {
            // NumPy: np.empty_like(a, shape=(4,5)) → shape=(4,5), dtype preserved from a
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, shape: new Shape(4, 5));
            r.shape.Should().BeEquivalentTo(new[] { 4, 5 });
            r.dtype.Should().Be(typeof(int), "dtype preserved from prototype");
        }

        [TestMethod]
        public void ShapeOverride_2DTo1D()
        {
            // NumPy: np.empty_like(a, shape=(10,)) → shape=(10,), dtype preserved
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, shape: new Shape(10));
            r.shape.Should().BeEquivalentTo(new[] { 10 });
            r.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void ShapeOverride_2DTo3D()
        {
            var a = np.arange(6).astype(np.float64).reshape(2, 3);
            var r = np.empty_like(a, shape: new Shape(2, 3, 4));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void ShapeOverride_WithDtypeOverride()
        {
            // NumPy: np.empty_like(a, dtype='float64', shape=(3,3,3)) → shape=(3,3,3), dtype=float64
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, typeof(double), new Shape(3, 3, 3));
            r.shape.Should().BeEquivalentTo(new[] { 3, 3, 3 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void ShapeOverride_SameSize()
        {
            // Override with same total element count but different shape
            var a = np.arange(12).astype(np.int32).reshape(3, 4);
            var r = np.empty_like(a, shape: new Shape(4, 3));
            r.shape.Should().BeEquivalentTo(new[] { 4, 3 });
            r.dtype.Should().Be(typeof(int));
            r.size.Should().Be(12);
        }

        [TestMethod]
        public void ShapeOverride_DifferentSize()
        {
            // Override with different total element count
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, shape: new Shape(10, 10));
            r.shape.Should().BeEquivalentTo(new[] { 10, 10 });
            r.dtype.Should().Be(typeof(int));
            r.size.Should().Be(100);
        }

        [TestMethod]
        public void ShapeOverride_ToScalar()
        {
            // NumPy: np.empty_like(a, shape=()) → shape=(), ndim=0
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, shape: Shape.Scalar);
            r.Shape.IsScalar.Should().BeTrue();
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void ShapeOverride_DefaultShape_UsesPrototype()
        {
            // When shape is default (empty), prototype shape is used
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            var r = np.empty_like(a, shape: default);
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
        }

        [TestMethod]
        public void ShapeOverride_FromBroadcast()
        {
            // Override shape on broadcast prototype
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(4, 3));
            var r = np.empty_like(b, shape: new Shape(5, 5));
            r.shape.Should().BeEquivalentTo(new[] { 5, 5 });
            r.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void ShapeOverride_FromSlice()
        {
            // Override shape on sliced prototype
            var a = np.arange(10);
            var s = a["2:7"];
            var r = np.empty_like(s, shape: new Shape(3, 3));
            r.shape.Should().BeEquivalentTo(new[] { 3, 3 });
            r.dtype.Should().Be(a.dtype);
        }

        // ==========================================================
        // 19. SHAPE DIMENSIONS INDEPENDENCE (aliasing fix)
        // ==========================================================

        [TestMethod]
        public void ShapeDimensionsArray_NotAliased()
        {
            // Fixed: np.empty_like now clones the dimensions int[] from prototype.
            // Previously shared the same reference (like full_like already did correctly).
            var a = np.arange(6).reshape(2, 3);
            var r = np.empty_like(a);

            object.ReferenceEquals(a.shape, r.shape).Should().BeFalse(
                "dimensions array should not be the same reference (defensive copy)");
        }

        [TestMethod]
        public void ShapeDimensionsArray_MutationIsolation()
        {
            // Verify that having separate int[] means mutations to one don't affect the other.
            // We test this by creating empty_like, then reshaping the original —
            // the result's shape should be unaffected.
            var a = np.arange(6).reshape(2, 3);
            var r = np.empty_like(a);

            // Reshape original to something different
            a = a.reshape(3, 2);

            // Result should still have original shape
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 },
                "result shape must be independent of prototype after creation");
        }
    }
}
