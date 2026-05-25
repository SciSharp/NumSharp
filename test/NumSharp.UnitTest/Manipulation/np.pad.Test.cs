using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_pad_Test : TestClass
    {
        // ============================== constant ==============================

        [TestMethod]
        public void Constant_Default_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, 2);
            r.array_equal(np.array(new long[] { 0, 0, 1, 2, 3, 4, 5, 0, 0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Constant_Scalar_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, 2, constant_values: 5L);
            r.array_equal(np.array(new long[] { 5, 5, 1, 2, 3, 4, 5, 5, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Constant_Pair_BeforeAfter()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), constant_values: (4L, 6L));
            r.array_equal(np.array(new long[] { 4, 4, 1, 2, 3, 4, 5, 6, 6, 6 })).Should().BeTrue();
        }

        [TestMethod]
        public void Constant_2D_PerAxis_PerSide()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.pad(arr, new int[,] { { 1, 2 }, { 3, 4 } },
                constant_values: new object[,] { { 9L, 8L }, { 7L, 6L } });
            // axis 0 left=9, right=8; axis 1 left=7, right=6 (corners from axis 0)
            r.shape.Should().BeEquivalentTo(new long[] { 5, 10 });
        }

        [TestMethod]
        public void Constant_2D_Scalar()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.pad(arr, 1);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 5 });
            // Surrounded by zeros, original in the centre.
            r.array_equal(np.array(new long[,] {
                { 0, 0, 0, 0, 0 },
                { 0, 1, 2, 3, 0 },
                { 0, 4, 5, 6, 0 },
                { 0, 0, 0, 0, 0 }
            })).Should().BeTrue();
        }

        [TestMethod]
        public void Constant_3D()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.pad(arr, 1, constant_values: -1L);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 5, 6 });
        }

        [TestMethod]
        public void Constant_ZeroPad_ReturnsCopy()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var r = np.pad(arr, 0);
            r.array_equal(arr).Should().BeTrue();
            ReferenceEquals(r, arr).Should().BeFalse();
        }

        // ============================== edge ==============================

        [TestMethod]
        public void Edge_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "edge");
            r.array_equal(np.array(new long[] { 1, 1, 1, 2, 3, 4, 5, 5, 5, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Edge_2D_CornerPropagation()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var r = np.pad(arr, 1, mode: "edge");
            r.array_equal(np.array(new long[,] {
                { 1, 1, 2, 3, 3 },
                { 1, 1, 2, 3, 3 },
                { 4, 4, 5, 6, 6 },
                { 4, 4, 5, 6, 6 }
            })).Should().BeTrue();
        }

        // ============================== reflect ==============================

        [TestMethod]
        public void Reflect_1D_Even()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "reflect");
            r.array_equal(np.array(new long[] { 3, 2, 1, 2, 3, 4, 5, 4, 3, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Reflect_1D_Odd()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "reflect", reflect_type: "odd");
            r.array_equal(np.array(new long[] { -1, 0, 1, 2, 3, 4, 5, 6, 7, 8 })).Should().BeTrue();
        }

        [TestMethod]
        public void Reflect_BigPad_Iterates()
        {
            // pad (5,5) > period 2 — must iterate
            var arr = np.array(new long[] { 1, 2, 3 });
            var r = np.pad(arr, (5, 5), mode: "reflect");
            r.array_equal(np.array(new long[] { 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Reflect_Singleton_FallsBackToEdge()
        {
            // axis_size == 1 → NumPy falls back to edge-fill.
            var arr = np.array(new long[,] { { 5 } });
            var r = np.pad(arr, 2, mode: "reflect");
            r.shape.Should().BeEquivalentTo(new long[] { 5, 5 });
            // Every cell == 5.
            for (int i = 0; i < r.size; i++)
            {
                var coord = new long[] { i / 5, i % 5 };
                r.GetInt64(coord).Should().Be(5);
            }
        }

        // ============================== symmetric ==============================

        [TestMethod]
        public void Symmetric_1D_Even()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "symmetric");
            r.array_equal(np.array(new long[] { 2, 1, 1, 2, 3, 4, 5, 5, 4, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Symmetric_1D_Odd()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "symmetric", reflect_type: "odd");
            r.array_equal(np.array(new long[] { 0, 1, 1, 2, 3, 4, 5, 5, 6, 7 })).Should().BeTrue();
        }

        // ============================== wrap ==============================

        [TestMethod]
        public void Wrap_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "wrap");
            r.array_equal(np.array(new long[] { 4, 5, 1, 2, 3, 4, 5, 1, 2, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Wrap_BigPad_Iterates()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var r = np.pad(arr, (5, 5), mode: "wrap");
            r.array_equal(np.array(new long[] { 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2 })).Should().BeTrue();
        }

        // ============================== empty ==============================

        [TestMethod]
        public void Empty_ShapeOnly_CenterMatches()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var r = np.pad(arr, 2, mode: "empty");
            r.shape.Should().BeEquivalentTo(new long[] { 7 });
            // Only center area is guaranteed to match.
            r.GetInt64(2).Should().Be(1);
            r.GetInt64(3).Should().Be(2);
            r.GetInt64(4).Should().Be(3);
        }

        // ============================== stat modes ==============================

        [TestMethod]
        public void Maximum_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "maximum");
            r.array_equal(np.array(new long[] { 5, 5, 1, 2, 3, 4, 5, 5, 5, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Maximum_StatLength()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "maximum", stat_length: 2);
            // first 2 = max(1,2) = 2; last 2 = max(4,5) = 5
            r.array_equal(np.array(new long[] { 2, 2, 1, 2, 3, 4, 5, 5, 5, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Minimum_1D()
        {
            var arr = np.array(new long[] { 3, 1, 4, 1, 5 });
            var r = np.pad(arr, 2, mode: "minimum");
            r.array_equal(np.array(new long[] { 1, 1, 3, 1, 4, 1, 5, 1, 1 })).Should().BeTrue();
        }

        [TestMethod]
        public void Mean_IntegerRoundsBankers()
        {
            // mean of [1,2,3,4,5] = 3.0 — exact
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, 2, mode: "mean");
            r.array_equal(np.array(new long[] { 3, 3, 1, 2, 3, 4, 5, 3, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Median_1D()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, 2, mode: "median");
            r.array_equal(np.array(new long[] { 3, 3, 1, 2, 3, 4, 5, 3, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Maximum_StatLength_Zero_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            Action act = () => np.pad(arr, 2, mode: "maximum", stat_length: 0);
            act.Should().Throw<ArgumentException>();
        }

        // ============================== linear_ramp ==============================

        [TestMethod]
        public void LinearRamp_DefaultEndZero()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "linear_ramp");
            r.array_equal(np.array(new long[] { 0, 0, 1, 2, 3, 4, 5, 3, 1, 0 })).Should().BeTrue();
        }

        [TestMethod]
        public void LinearRamp_AsymmetricEnds()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var r = np.pad(arr, (2, 3), mode: "linear_ramp", end_values: (5L, -4L));
            r.array_equal(np.array(new long[] { 5, 3, 1, 2, 3, 4, 5, 2, -1, -4 })).Should().BeTrue();
        }

        [TestMethod]
        public void LinearRamp_Float()
        {
            var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var r = np.pad(arr, (2, 3), mode: "linear_ramp", end_values: (0.0, 0.0));
            // exact float arithmetic
            r.shape.Should().BeEquivalentTo(new long[] { 10 });
            r.GetDouble(0).Should().Be(0.0);
            r.GetDouble(1).Should().Be(0.5);
            r.GetDouble(2).Should().Be(1.0);
            r.GetDouble(9).Should().Be(0.0);
        }

        // ============================== callable ==============================

        [TestMethod]
        public void Callable_NumPyDocExample()
        {
            // From NumPy docs:
            //   def pad_with(vector, pad_width, iaxis, kwargs):
            //       pad_value = kwargs.get('padder', 10)
            //       vector[:pad_width[0]] = pad_value
            //       vector[-pad_width[1]:] = pad_value
            //   np.pad(np.arange(6).reshape(2,3), 2, pad_with)
            // → 6x7 array with center [[0,1,2],[3,4,5]] surrounded by 10s

            void PadFunc(NDArray vec, (int before, int after) pw, int axis, object kwargs)
            {
                long pv = 10L;
                using var scalarBefore = NDArray.Scalar(pv, vec.GetTypeCode);
                using var scalarAfter = NDArray.Scalar(pv, vec.GetTypeCode);
                vec[new Slice(null, pw.before)] = scalarBefore;
                vec[new Slice(-pw.after, null)] = scalarAfter;
            }

            var a = np.arange(6).reshape(2, 3);
            var r = np.pad(a, 2, PadFunc, null);
            r.shape.Should().BeEquivalentTo(new long[] { 6, 7 });
            // Center
            r.GetInt64(new long[] { 2, 2 }).Should().Be(0);
            r.GetInt64(new long[] { 2, 4 }).Should().Be(2);
            r.GetInt64(new long[] { 3, 2 }).Should().Be(3);
            r.GetInt64(new long[] { 3, 4 }).Should().Be(5);
            // Borders
            r.GetInt64(new long[] { 0, 0 }).Should().Be(10);
            r.GetInt64(new long[] { 5, 6 }).Should().Be(10);
        }

        // ============================== pad_width shapes ==============================

        [TestMethod]
        public void PadWidth_Scalar_Broadcasts()
        {
            var arr = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var r = np.pad(arr, 1);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 4 });
        }

        [TestMethod]
        public void PadWidth_Tuple_Broadcasts()
        {
            var arr = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var r = np.pad(arr, (1, 2));
            r.shape.Should().BeEquivalentTo(new long[] { 5, 5 });
        }

        [TestMethod]
        public void PadWidth_PerAxis_Rectangular()
        {
            var arr = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var r = np.pad(arr, new int[,] { { 1, 0 }, { 0, 2 } });
            r.shape.Should().BeEquivalentTo(new long[] { 3, 4 });
        }

        [TestMethod]
        public void PadWidth_Dict_PerAxis()
        {
            var arr = np.arange(6).reshape(2, 3);
            var pw = new System.Collections.Generic.Dictionary<int, object> { { 1, (1, 2) } };
            var r = np.pad(arr, pw);
            r.shape.Should().BeEquivalentTo(new long[] { 2, 6 });
        }

        [TestMethod]
        public void PadWidth_Dict_NegativeAxis()
        {
            var arr = np.arange(6).reshape(2, 3);
            var pw = new System.Collections.Generic.Dictionary<int, object> { { -1, 2 } };
            var r = np.pad(arr, pw);
            r.shape.Should().BeEquivalentTo(new long[] { 2, 7 });
        }

        // ============================== error semantics ==============================

        [TestMethod]
        public void Negative_PadWidth_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            Action act = () => np.pad(arr, -1);
            act.Should().Throw<ArgumentException>().WithMessage("*negative*");
        }

        [TestMethod]
        public void UnknownMode_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            Action act = () => np.pad(arr, 1, mode: "fancy");
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void ReflectType_Invalid_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            Action act = () => np.pad(arr, 1, mode: "reflect", reflect_type: "weird");
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void UnsupportedKwarg_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            // constant_values is for 'constant', not 'edge'
            Action act = () => np.pad(arr, 1, mode: "edge", constant_values: 5);
            act.Should().Throw<ArgumentException>();
        }

        // ============================== dtype coverage ==============================

        [TestMethod]
        public void Dtype_Byte_Constant()
        {
            var arr = np.array(new byte[] { 1, 2, 3 });
            var r = np.pad(arr, 1, constant_values: (byte)9);
            r.array_equal(np.array(new byte[] { 9, 1, 2, 3, 9 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Int32_Reflect()
        {
            var arr = np.array(new int[] { 1, 2, 3 });
            var r = np.pad(arr, 1, mode: "reflect");
            r.array_equal(np.array(new int[] { 2, 1, 2, 3, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Double_Mean()
        {
            var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var r = np.pad(arr, 2, mode: "mean");
            r.GetDouble(0).Should().Be(3.0);
            r.GetDouble(8).Should().Be(3.0);
        }

        [TestMethod]
        public void Dtype_Float_Edge()
        {
            var arr = np.array(new float[] { 1f, 2f, 3f, 4f, 5f });
            var r = np.pad(arr, 1, mode: "edge");
            r.GetSingle(0).Should().Be(1f);
            r.GetSingle(6).Should().Be(5f);
        }
    }
}
