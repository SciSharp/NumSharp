using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The bridge inside NumSharp's ambient disposal scope, <see cref="NDScope"/>. A scope tracks EVERY NDArray
    ///     constructed on its thread while it is open, the bridge's own temporaries included (the Single conversion of a
    ///     Half array and the Double conversion of a Decimal array, both owned by the view), and disposes each one not
    ///     yielded with <see cref="NDScope.Returns{T}(T)"/> when it closes. A view that leaves the scope must keep reading
    ///     the right values after that, also through ML.NET's lazy transforms. It does because the view pins its buffer
    ///     with a counted reference of its own, separate from the NDArray's: the scope's dispose drops only the NDArray's.
    /// </summary>
    /// <remarks>
    ///     Each test hands the freed memory out again before it reads (<see cref="Churn"/>), so a view left reading a
    ///     buffer the scope freed reads 0x55 bytes instead of the values. Under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c> a freed
    ///     buffer is unmapped instead, and the read ends the test host. Training and scoring inside a scope construct no
    ///     NDArray per row: the cursor reads the buffer through its strides (5,000 rows of SDCA: the scope's list did not
    ///     grow; scratch probe <c>probes/ndscope_bridges.cs</c> of the release QA).
    /// </remarks>
    [TestClass]
    public class NDScopeTests : MLNetTestBase
    {
        /// <summary>The values every test exports: exact in float16, float32, float64 and Decimal, and none of them 0x55 bytes.</summary>
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
        ///     A Half array crosses as a Single conversion the view owns, and that conversion is an NDArray the open scope
        ///     tracks. The scope disposes it when it closes; the view that left the scope still reads the values.
        /// </summary>
        [TestMethod]
        public void HalfView_LeavesTheScope_KeepsItsConversion()
        {
            NDArrayDataView view;
            using (NDScope.Open())
            {
                NDArray half = np.array(Values).astype(np.float16).reshape(2, 4);
                view = half.AsDataView("x");
            }

            using (view)
            {
                Churn(Values.Length * sizeof(float));
                using NDArray back = view.ToNDArray("x");
                AssertShape(back, 2, 4);
                back.ToArray<float>().Should().Equal(Values.Select(v => (float)v));
            }
        }

        /// <summary>
        ///     A Decimal array, one scalar column per feature: the Double conversion the view owns, constructed and later
        ///     disposed by the scope.
        /// </summary>
        [TestMethod]
        public void DecimalScalarColumns_LeaveTheScope_KeepTheirConversion()
        {
            NDArrayDataView view;
            using (NDScope.Open())
            {
                NDArray source = np.array(Values.Select(v => (decimal)v).ToArray()).reshape(4, 2);
                view = source.AsDataView(new[] { "a", "b" });
            }

            using (view)
            {
                Churn(Values.Length * sizeof(double));
                using NDArray a = view.ToNDArray("a");
                using NDArray b = view.ToNDArray("b");
                a.ToArray<double>().Should().Equal(Values.Where((_, i) => i % 2 == 0));
                b.ToArray<double>().Should().Equal(Values.Where((_, i) => i % 2 == 1));
            }
        }

        /// <summary>
        ///     Zero-copy: the SOURCE array is constructed inside the scope, so the scope disposes it when it closes, and a
        ///     lazy transform's output built inside the scope is read only after it. The view's pin keeps the buffer valid
        ///     for both reads.
        /// </summary>
        [TestMethod]
        public void ZeroCopyView_OfAnArrayTheScopeDisposes_ReadsLazilyAfterTheScope()
        {
            float[] values = Values.Select(v => (float)v).ToArray();
            NDArrayDataView view;
            IDataView scaled;
            NDArray source;
            using (NDScope.Open())
            {
                source = np.array(values).reshape(4, 2);
                view = source.AsDataView("x");
                scaled = Ml.Transforms.NormalizeMinMax("x").Fit(view).Transform(view);
            }

            using (view)
            {
                source.IsDisposed.Should().BeTrue("the scope disposed the array it tracked");
                Churn(values.Length * sizeof(float));
                using NDArray raw = view.ToNDArray("x");
                raw.ToArray<float>().Should().Equal(values);
                using NDArray normalized = scaled.ToNDArray("x");
                AssertShape(normalized, 4, 2);
                // NormalizeMinMax (fixZero, its default) divides each column by its largest magnitude: 7 for the first
                // column, 8.5 for the second. A read of freed memory would have scaled 0x55 garbage instead.
                float[] got = normalized.ToArray<float>();
                for (int i = 0; i < got.Length; i++)
                    got[i].Should().BeApproximately(values[i] / (i % 2 == 0 ? 7f : 8.5f), 1e-6f, $"element {i}");
            }
        }

        /// <summary>
        ///     A zero-copy VBuffer import (<see cref="NDArrayMLNetInterop.AsNDArray{T}(VBuffer{T})"/>) left to the scope:
        ///     the scope disposes the lease when it closes, which releases the pin on the VBuffer's array, so the import
        ///     counter is back at its baseline the moment the scope is gone.
        /// </summary>
        [TestMethod]
        public void VBufferImport_LeftToTheScope_IsReleasedWhenItCloses()
        {
            int baseline = NDArrayMLNetInterop.LiveImports;
            float[] values = Values.Select(v => (float)v).ToArray();
            var buffer = new VBuffer<float>(values.Length, values);
            NDArray lease;
            using (NDScope.Open())
            {
                lease = buffer.AsNDArray<float>();
                lease.ToArray<float>().Should().Equal(values);
                NDArrayMLNetInterop.LiveImports.Should().Be(baseline + 1);
            }

            lease.IsDisposed.Should().BeTrue();
            NDArrayMLNetInterop.LiveImports.Should().Be(baseline);
        }
    }
}
