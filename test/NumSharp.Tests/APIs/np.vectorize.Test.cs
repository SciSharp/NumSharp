using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     Tests for <see cref="np.vectorize"/> / <see cref="np.frompyfunc"/> and the <see cref="Vectorized"/>
    ///     object, verified against NumPy 2.4.2 output. Covers both modes (element-wise via the fused
    ///     <c>NDExpr.Call</c> path and gufunc signature via per-slice iteration), multi-output, otypes,
    ///     broadcasting, dtype behaviour, non-contiguous inputs, and the error parity.
    ///
    ///     NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    ///     NumPy source: numpy/lib/_function_base_impl.py (vectorize), numpy/_core/src/umath/umathmodule.c (frompyfunc)
    /// </summary>
    [TestClass]
    public class np_vectorize_Test
    {
        // ---- element-wise mode (typed Func-returning overloads) ----

        /// <summary>The docstring example: myfunc(a,b)=a>b?a-b:a+b, vectorized over an array and a scalar → [3 4 1 2].</summary>
        [TestMethod]
        public void Elementwise_Basic_ScalarBroadcast()
        {
            var vf = np.vectorize((int a, int b) => a > b ? a - b : a + b);
            var r = vf(np.array(new[] { 1, 2, 3, 4 }), 2);
            r.Should().BeShaped(4);
            r.dtype.Should().Be(np.int32);   // C# int is int32 (NumPy's Python int is int64 — language difference)
            r.Should().BeOfValues(3, 4, 1, 2);
        }

        /// <summary>A bool-returning delegate yields a bool array (dtype from the return type, no otypes needed).</summary>
        [TestMethod]
        public void Elementwise_BoolReturn()
        {
            var vf = np.vectorize((int x) => x > 2);
            var r = vf(np.array(new[] { 1, 2, 3, 4 }));
            r.dtype.Should().Be(np.bool_);
            r.Should().BeOfValues(false, false, true, true);
        }

        /// <summary>The output dtype is the delegate's return type: an int-returning func gives an int32 array.</summary>
        [TestMethod]
        public void Elementwise_ReturnTypeFixesDtype()
        {
            var vf = np.vectorize((int x) => x * 2);
            var r = vf(np.array(new[] { 1, 2, 3 }));
            r.dtype.Should().Be(np.int32);
            r.Should().BeOfValues(2, 4, 6);
        }

        /// <summary>Two inputs of different dtypes broadcast and promote per NEP50 — int + double → double.</summary>
        [TestMethod]
        public void Elementwise_MixedDtypeArgs()
        {
            var vf = np.vectorize((int a, double b) => a + b);
            var r = vf(np.array(new[] { 1, 2, 3 }), np.array(new[] { 0.5, 0.5, 0.5 }));
            r.dtype.Should().Be(np.float64);
            r.Should().BeOfValues(1.5, 2.5, 3.5);
        }

        /// <summary>Two array inputs broadcast together: (3,1) + (1,4) → (3,4).</summary>
        [TestMethod]
        public void Elementwise_Broadcast()
        {
            var vf = np.vectorize((long a, long b) => a + b);
            var r = vf(np.arange(3).reshape(3, 1), np.arange(4).reshape(1, 4));
            r.Should().BeShaped(3, 4);
            r.Should().BeOfValues(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5);
        }

        /// <summary>A 0-d input yields a 0-d result (NumPy's scalar-in / scalar-out).</summary>
        [TestMethod]
        public void Elementwise_ScalarInput()
        {
            var vf = np.vectorize((int x) => x * 2);
            var r = vf(NDArray.Scalar(5));
            r.Should().BeShaped();
            r.GetInt32(0).Should().Be(10);
        }

        /// <summary>
        ///     An empty input works WITHOUT otypes — the C# return type fixes the dtype, so unlike NumPy
        ///     (which raises "cannot call vectorize on size 0 inputs unless otypes is set") the empty result is
        ///     allocated at the delegate's return dtype.
        /// </summary>
        [TestMethod]
        public void Elementwise_EmptyInput_NoOtypesNeeded()
        {
            var vf = np.vectorize((int x) => x * 2);
            var r = vf(np.array(new int[0]));
            r.Should().BeShaped(0);
            r.dtype.Should().Be(np.int32);
        }

        /// <summary>Element-wise reads through any layout: a transposed (non-contiguous) input applies correctly.</summary>
        [TestMethod]
        public void Elementwise_NonContiguousInput()
        {
            var m = np.arange(6).reshape(2, 3).T;   // (3,2) strided view
            var vf = np.vectorize((long x) => x * 2);
            var r = vf(m);
            r.Should().BeShaped(3, 2);
            r.Should().BeOfValues(0, 6, 2, 8, 4, 10);
        }

        /// <summary>otypes overrides the output dtype: an int-returning func cast to float64 → [2. 4. 6.].</summary>
        [TestMethod]
        public void Elementwise_OtypesOverride()
        {
            var vf = np.vectorize((int x) => x * 2, new DType[] { np.float64 });
            var r = vf(np.array(new[] { 1, 2, 3 }));
            r.dtype.Should().Be(np.float64);
            r.Should().BeOfValues(2.0, 4.0, 6.0);
        }

        /// <summary>The same vectorized delegate can be called repeatedly (a reusable object under the hood).</summary>
        [TestMethod]
        public void Elementwise_Reusable()
        {
            var vf = np.vectorize((int x) => x + 1);
            vf(np.array(new[] { 1, 2 })).Should().BeOfValues(2, 3);
            vf(np.array(new[] { 10, 20, 30 })).Should().BeOfValues(11, 21, 31);
        }

        // ---- element-wise multi-output (tuple-returning) ----

        /// <summary>A 2-tuple-returning delegate binds the multi-output overload → a Vectorized with nout=2.</summary>
        [TestMethod]
        public void Elementwise_MultiOutput_TwoComponents()
        {
            var vmm = np.vectorize((int x) => (x + 1, x - 1));
            vmm.nout.Should().Be(2);
            vmm.nin.Should().Be(1);
            var r = vmm.CallMany(np.array(new[] { 10, 20, 30 }));
            r.Length.Should().Be(2);
            r[0].Should().BeOfValues(11, 21, 31);
            r[1].Should().BeOfValues(9, 19, 29);
        }

        /// <summary>Multi-output over two inputs broadcasts them, then produces the tuple components.</summary>
        [TestMethod]
        public void Elementwise_MultiOutput_TwoInputs()
        {
            var v = np.vectorize((long a, long b) => (a + b, a - b));
            var r = v.CallMany(np.array(new[] { 5L, 6L }), np.array(new[] { 1L, 2L }));
            r[0].Should().BeOfValues(6, 8);
            r[1].Should().BeOfValues(4, 4);
        }

        /// <summary>Calling <see cref="Vectorized.Call"/> on a multi-output function is a directed error.</summary>
        [TestMethod]
        public void Elementwise_MultiOutput_CallThrows()
        {
            var vmm = np.vectorize((int x) => (x + 1, x - 1));
            ((Action)(() => vmm.Call(np.array(new[] { 1, 2, 3 }))))
                .Should().Throw<InvalidOperationException>().WithMessage("*CallMany*");
        }

        // ---- signature (gufunc) mode ----

        /// <summary>signature "(n)->()" reduces each 1-D slice: sum of each row of a (2,3) → [3 12].</summary>
        [TestMethod]
        public void Signature_ReduceRows()
        {
            var vsum = np.vectorize((NDArray x) => np.sum(x), "(n)->()");
            var r = vsum.Call(np.arange(6).reshape(2, 3));
            r.Should().BeShaped(2);
            r.Should().BeOfValues(3L, 12L);
        }

        /// <summary>signature "(m,n),(n)->(m)" is a batched matrix-vector product.</summary>
        [TestMethod]
        public void Signature_Matvec()
        {
            var vmv = np.vectorize((NDArray A, NDArray x) => np.matmul(A, x), "(m,n),(n)->(m)");
            var r = vmv.Call(np.arange(6).reshape(2, 3).astype(np.float64), np.array(new[] { 1.0, 1.0, 1.0 }));
            r.Should().BeShaped(2);
            r.Should().BeOfValues(3.0, 12.0);
        }

        /// <summary>signature "(n),(m)->(k)" introduces a new output dimension (k), sized from the first result.</summary>
        [TestMethod]
        public void Signature_NewOutputDim_Convolve()
        {
            var vc = np.vectorize((NDArray a, NDArray v) => np.convolve(a, v), "(n),(m)->(k)");
            var r = vc.Call(np.eye(4), np.array(new[] { 1.0, 2.0, 1.0 }));
            r.Should().BeShaped(4, 6);
            r[0].Should().BeOfValues(1.0, 2.0, 1.0, 0.0, 0.0, 0.0);
        }

        /// <summary>A multi-output signature "(n)->(),()" returns two arrays (min and max per row).</summary>
        [TestMethod]
        public void Signature_MultiOutput()
        {
            var v = np.vectorize((Delegate)(Func<NDArray, (NDArray, NDArray)>)(x => (np.min(x), np.max(x))), "(n)->(),()");
            var r = v.CallMany(np.arange(6).reshape(2, 3));
            r.Length.Should().Be(2);
            r[0].Should().BeOfValues(0L, 3L);   // min
            r[1].Should().BeOfValues(2L, 5L);   // max
        }

        /// <summary>Signature mode broadcasts the non-core dimensions: a (1,4) key against two rows → 2 dot products.</summary>
        [TestMethod]
        public void Signature_BroadcastNonCore()
        {
            var vdot = np.vectorize((NDArray a, NDArray b) => np.dot(a, b), "(n),(n)->()");
            var r = vdot.Call(np.array(new long[,] { { 0, 1, 2, 3 } }),
                              np.array(new long[,] { { 1, 2, 3, 4 }, { 4, 3, 2, 1 } }));
            r.Should().BeShaped(2);
            r.Should().BeOfValues(20L, 10L);
        }

        // ---- frompyfunc (typed sibling) ----

        /// <summary>frompyfunc builds a broadcasting callable with typed output (not NumPy's object arrays).</summary>
        [TestMethod]
        public void Frompyfunc_Basic()
        {
            var sq = np.frompyfunc((Func<int, int>)(x => x * x), 1, 1);
            sq.nin.Should().Be(1);
            sq.nout.Should().Be(1);
            sq.Call(np.array(new[] { 2, 3, 4 })).Should().BeOfValues(4, 9, 16);
        }

        /// <summary>frompyfunc validates nin against the delegate's actual parameter count (C# knows the arity).</summary>
        [TestMethod]
        public void Frompyfunc_NinMismatch_Throws()
        {
            ((Action)(() => np.frompyfunc((Func<int, int>)(x => x), 2, 1)))
                .Should().Throw<ValueError>().WithMessage("*nin*");
        }

        /// <summary>frompyfunc validates nout against the delegate's return-component count.</summary>
        [TestMethod]
        public void Frompyfunc_NoutMismatch_Throws()
        {
            ((Action)(() => np.frompyfunc((Func<int, int>)(x => x), 1, 2)))
                .Should().Throw<ValueError>().WithMessage("*nout*");
        }

        // ---- error parity ----

        /// <summary>A malformed gufunc signature raises NumPy's verbatim "not a valid gufunc signature" ValueError.</summary>
        [TestMethod]
        public void Error_BadSignature()
        {
            ((Action)(() => np.vectorize((NDArray x) => np.sum(x), "(n)-()")))
                .Should().Throw<ValueError>().WithMessage("not a valid gufunc signature*");
        }

        /// <summary>A signature whose input arity differs from the delegate's is rejected at construction.</summary>
        [TestMethod]
        public void Error_SignatureArityMismatch()
        {
            ((Action)(() => np.vectorize((NDArray x) => np.sum(x), "(n),(m)->()")))
                .Should().Throw<ValueError>().WithMessage("*input argument*");
        }

        /// <summary>An otypes length that disagrees with the output count is rejected.</summary>
        [TestMethod]
        public void Error_OtypesLengthMismatch()
        {
            ((Action)(() => np.vectorize((int x) => x * 2, new DType[] { np.float64, np.int32 })))
                .Should().Throw<ValueError>().WithMessage("*otypes*");
        }

        /// <summary>An NDArray-typed parameter without a signature is a directed error (element-wise needs scalar types).</summary>
        [TestMethod]
        public void Error_NDArrayParamWithoutSignature()
        {
            ((Action)(() => np.vectorize((Delegate)(Func<NDArray, NDArray>)(x => x))))
                .Should().Throw<ArgumentException>().WithMessage("*signature*");
        }

        /// <summary>Signature mode with a delegate returning fewer outputs than the signature declares raises.</summary>
        [TestMethod]
        public void Error_SignatureOutputCountMismatch()
        {
            var v = np.vectorize((NDArray x) => np.min(x), "(n)->(),()");
            ((Action)(() => v.CallMany(np.arange(6).reshape(2, 3))))
                .Should().Throw<ValueError>().WithMessage("wrong number of outputs*");
        }

        /// <summary>A named core dimension with inconsistent sizes across inputs raises NumPy's verbatim message.</summary>
        [TestMethod]
        public void Error_InconsistentCoreDim()
        {
            var v = np.vectorize((NDArray a, NDArray b) => np.dot(a, b), "(n),(n)->()");
            ((Action)(() => v.Call(np.arange(3), np.arange(4))))
                .Should().Throw<ValueError>().WithMessage("inconsistent size for core dimension*");
        }

        /// <summary>Element-wise inputs that do not broadcast raise the shape error listing every operand.</summary>
        [TestMethod]
        public void Error_BroadcastMismatch()
        {
            var v = np.vectorize((long a, long b) => a + b);
            ((Action)(() => v(np.arange(3), np.arange(4))))
                .Should().Throw<IncorrectShapeException>();
        }

        /// <summary>Signature mode on size-0 input with no otypes raises NumPy's verbatim size-0 message.</summary>
        [TestMethod]
        public void Error_SignatureEmptyInput_NeedsOtypes()
        {
            var v = np.vectorize((NDArray a, NDArray b) => np.convolve(a, b), "(n),(m)->(k)");
            ((Action)(() => v.Call(np.zeros(new Shape(0, 4), np.float64), np.array(new[] { 1.0, 2.0 }))))
                .Should().Throw<ValueError>().WithMessage("cannot call `vectorize` on size 0 inputs*");
        }

        /// <summary>A null delegate is rejected.</summary>
        [TestMethod]
        public void Error_NullDelegate()
        {
            ((Action)(() => np.vectorize((Delegate)null)))
                .Should().Throw<ArgumentNullException>();
        }

        /// <summary>Passing the wrong number of input arrays to a callable is rejected.</summary>
        [TestMethod]
        public void Error_WrongArgCount()
        {
            var v = np.vectorize((Delegate)(Func<int, int, int>)((a, b) => a + b));
            ((Action)(() => v.Call(np.array(new[] { 1, 2 }))))
                .Should().Throw<ArgumentException>().WithMessage("*expected 2 input*");
        }
    }
}
