using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The bridge inside NumSharp's ambient disposal scope, <see cref="NDScope"/>. A scope tracks EVERY NDArray
    ///     constructed on its thread while it is open, the bridge's own temporaries included (the float64 conversion of a
    ///     Decimal array, the C-order copy of a strided view, <c>Run</c>'s cast inputs and its outputs), and disposes each
    ///     one not yielded with <see cref="NDScope.Returns{T}(T)"/> when it closes. A handle, or an output, that leaves the
    ///     scope must keep reading the right bytes after that. It does because a handle holds a counted reference on its
    ///     buffer of its own, separate from the temporary NDArray's: the scope's dispose drops only the temporary's.
    /// </summary>
    /// <remarks>
    ///     Each test hands the freed memory out again before it reads (<see cref="Churn"/>), so a handle left pointing at a
    ///     buffer the scope freed reads 0x55 bytes instead of the values. Under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c> a freed
    ///     buffer is unmapped instead, and the read ends the test host.
    /// </remarks>
    [TestClass]
    public class NDScopeTests : OnnxTestBase
    {
        /// <summary>The values every test exports: exact in float32, float64 and Decimal, and none of them 0x55 bytes.</summary>
        private static readonly double[] Values = { 1.5, -2.25, 3, 4.125, -0.5, 6.75, 7, 8.5 };

        /// <summary>
        ///     Takes buffers of <paramref name="bytes"/> and of the sizes around it from the pool, fills them with 0x55 and
        ///     gives them back: a buffer freed just before is handed out again and overwritten (1.2e103 as float64, 1.5e13
        ///     as float32), so a reader still pointing at it sees garbage instead of a stale copy of the right values.
        /// </summary>
        /// <param name="bytes">The byte size of the buffer that must NOT have been freed.</param>
        private static void Churn(long bytes)
        {
            var held = new List<NDArray>();
            foreach (long size in new[] { bytes, bytes * 2, Math.Max(64, bytes) })
                for (int i = 0; i < 8; i++)
                    held.Add(np.full(new Shape(size), (byte)0x55, np.uint8));
            foreach (NDArray buffer in held)
                buffer.Dispose();
        }

        /// <summary>
        ///     Runs the float64 identity model on <paramref name="value"/> through ONNX Runtime's own <c>Run</c>, so the
        ///     bridge reads nothing: what comes back is what the native tensor's memory holds.
        /// </summary>
        /// <param name="value">A float64 tensor.</param>
        /// <returns>The model's output, copied out of ORT's memory.</returns>
        private static double[] RunIdentity64(OrtValue value)
        {
            using InferenceSession session = OpenSession("identity_float64");
            using var runOptions = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, new[] { "X" }, new[] { value }, new[] { "Y" });
            return outputs[0].GetTensorDataAsSpan<double>().ToArray();
        }

        /// <summary>
        ///     A Decimal array crosses as a float64 conversion the handle owns, and that conversion is an NDArray the open
        ///     scope tracks. The scope disposes it when it closes; the handle that left the scope still feeds the values.
        /// </summary>
        [TestMethod]
        public void DecimalHandle_LeavesTheScope_KeepsItsConversion()
        {
            OrtTensor handle;
            using (NDScope.Open())
            {
                NDArray source = np.array(Values.Select(v => (decimal)v).ToArray());
                handle = source.AsOrtValue();
            }

            using (handle)
            {
                Churn(Values.Length * sizeof(double));
                handle.IsView.Should().BeFalse("a Decimal array crosses as a copy");
                RunIdentity64(handle.Value).Should().Equal(Values);
                handle.Value.GetTensorDataAsSpan<double>().ToArray().Should().Equal(Values);
            }
        }

        /// <summary>
        ///     A transposed Decimal array is converted (in 'K' order, a Fortran-ordered result here) and then laid out in C
        ///     order: two temporaries the scope tracks, the second one owned by the handle.
        /// </summary>
        [TestMethod]
        public void TransposedDecimalHandle_LeavesTheScope_KeepsItsCOrderCopy()
        {
            OrtTensor handle;
            double[] expected;
            using (NDScope.Open())
            {
                NDArray source = np.array(Values.Select(v => (decimal)v).ToArray()).reshape(2, 4).T;
                expected = np.ascontiguousarray(source.astype(np.float64)).ToArray<double>();
                handle = source.AsOrtValue();
            }

            using (handle)
            {
                Churn(Values.Length * sizeof(double));
                RunIdentity64(handle.Value).Should().Equal(expected);
            }
        }

        /// <summary>
        ///     A strided float32 view is fed as a C-order copy the handle owns (<see cref="OnnxCodecMode.Auto"/>); the copy
        ///     is constructed, and later disposed, by the scope.
        /// </summary>
        [TestMethod]
        public void StridedCopyHandle_LeavesTheScope_KeepsItsCopy()
        {
            float[] doubled = Values.Concat(Values).Select(v => (float)v).ToArray();
            OrtTensor handle;
            using (NDScope.Open())
            {
                NDArray strided = np.array(doubled)["::2"];
                handle = strided.AsOrtValue();
            }

            using (handle)
            {
                Churn(Values.Length * sizeof(float));
                handle.IsView.Should().BeFalse("a strided view crosses as a C-order copy");
                using InferenceSession session = OpenSession("identity_float32");
                using var runOptions = new RunOptions();
                using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, new[] { "X" }, new[] { handle.Value }, new[] { "Y" });
                outputs[0].GetTensorDataAsSpan<float>().ToArray().Should().Equal(doubled.Where((_, i) => i % 2 == 0));
            }
        }

        /// <summary>
        ///     Zero-copy: the SOURCE array is constructed inside the scope, so the scope disposes it when it closes. The
        ///     handle's own reference keeps the buffer, and the tensor over it, valid.
        /// </summary>
        [TestMethod]
        public void ZeroCopyHandle_OfAnArrayTheScopeDisposes_StaysValid()
        {
            float[] values = Values.Select(v => (float)v).ToArray();
            OrtTensor handle;
            NDArray source;
            using (NDScope.Open())
            {
                source = np.array(values);
                handle = source.AsOrtValue();
            }

            using (handle)
            {
                source.IsDisposed.Should().BeTrue("the scope disposed the array it tracked");
                handle.IsView.Should().BeTrue();
                Churn(values.Length * sizeof(float));
                using InferenceSession session = OpenSession("identity_float32");
                using var runOptions = new RunOptions();
                using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, new[] { "X" }, new[] { handle.Value }, new[] { "Y" });
                outputs[0].GetTensorDataAsSpan<float>().ToArray().Should().Equal(values);
            }
        }

        /// <summary>
        ///     The legacy tier: <see cref="NDArrayOnnxInterop.AsDenseTensor{T}"/> of a Decimal array (the float64 conversion
        ///     again, behind a <c>DenseTensor</c>), and the handle's <see cref="OrtTensor.Source"/>, which is that
        ///     conversion: the scope disposed the NDArray (<c>IsDisposed</c>), and it still reads, because the handle holds
        ///     the buffer.
        /// </summary>
        [TestMethod]
        public void DenseTensorHandle_OfADecimalArray_LeavesTheScope()
        {
            OrtTensor<double> handle;
            using (NDScope.Open())
            {
                NDArray source = np.array(Values.Select(v => (decimal)v).ToArray());
                handle = source.AsDenseTensor<double>();
            }

            using (handle)
            {
                Churn(Values.Length * sizeof(double));
                handle.Tensor.ToArray().Should().Equal(Values);
                handle.Source.IsDisposed.Should().BeTrue("the conversion was constructed inside the scope, which disposed it");
                handle.Source.ToArray<double>().Should().Equal(Values);
            }
        }

        /// <summary>
        ///     <c>Run</c> inside a scope with an input that needs a cast (int16 to the model's float32, 'safe'): the bridge
        ///     disposes its cast temporary and the scope disposes it again (a no-op, <see cref="NDArray.Dispose"/> is
        ///     idempotent). The outputs yielded with <see cref="NDScope.Returns{T}(T)"/> survive the scope: a copied one, a
        ///     zero-copy one over ORT's memory, and the zero-copy output that IS the fed input (passthrough's X), which
        ///     <c>Run</c> returns as a view over the cast temporary.
        /// </summary>
        [TestMethod]
        public void Run_InAScope_YieldedOutputsSurvive_TheFedInputViewIncluded()
        {
            float[] fed = { 1, -2, 3, 4, -5, 6, 7, 8 };
            using InferenceSession session = OpenSession("passthrough");
            NDArray copiedY, viewedX, viewedY, leftToTheScope;
            using (NDScope scope = NDScope.Open())
            {
                NDArray ints = np.array(new short[] { 1, -2, 3, 4, -5, 6, 7, 8 });
                IReadOnlyDictionary<string, NDArray> copied = session.Run(new Dictionary<string, NDArray> { ["X"] = ints });
                IReadOnlyDictionary<string, NDArray> viewed = session.Run(new Dictionary<string, NDArray> { ["X"] = ints }, zeroCopyOutputs: true);
                copiedY = scope.Returns(copied["Y"]);
                viewedX = scope.Returns(viewed["X"]);
                viewedY = scope.Returns(viewed["Y"]);
                leftToTheScope = copied["X"];
            }

            using (copiedY)
            using (viewedX)
            using (viewedY)
            {
                leftToTheScope.IsDisposed.Should().BeTrue("an output not yielded is the scope's to dispose");
                Churn(fed.Length * sizeof(float));
                copiedY.ToArray<float>().Should().Equal(fed.Select(v => -v));
                viewedX.ToArray<float>().Should().Equal(fed);
                viewedY.ToArray<float>().Should().Equal(fed.Select(v => -v));
            }
        }

        /// <summary>
        ///     A zero-copy import (<see cref="NDArrayOnnxInterop.AsNDArray(OrtValue, bool, OnnxCodecMode)"/> owning its
        ///     value) left to the scope: the scope disposes the lease when it closes, which releases ONNX Runtime's value,
        ///     so the import counter is back at its baseline the moment the scope is gone.
        /// </summary>
        [TestMethod]
        public void ZeroCopyImport_LeftToTheScope_IsReleasedWhenItCloses()
        {
            int baseline = NDArrayOnnxInterop.LiveImports;
            using InferenceSession session = OpenSession("identity_float64");
            NDArray lease;
            using (NDScope.Open())
            {
                using OrtValue input = np.array(Values).ToOrtValue();
                using var runOptions = new RunOptions();
                IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, new[] { "X" }, new[] { input }, new[] { "Y" });
                lease = outputs[0].AsNDArray(ownsValue: true);
                lease.ToArray<double>().Should().Equal(Values);
                NDArrayOnnxInterop.LiveImports.Should().Be(baseline + 1);
            }

            lease.IsDisposed.Should().BeTrue();
            NDArrayOnnxInterop.LiveImports.Should().Be(baseline);
        }
    }
}
