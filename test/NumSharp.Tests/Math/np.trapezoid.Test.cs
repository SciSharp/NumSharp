using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.trapezoid — composite trapezoidal integration along an axis. All expected values are
    ///     from NumPy 2.4.2 (probed directly). The result is a pure composition
    ///     <c>sum(d*(y[1:]+y[:-1])/2, axis)</c>, so it is BIT-exact with NumPy for finite data; these
    ///     assertions cover the docstring examples, the dtype tier (NEP50), scalar/coordinate spacing,
    ///     the axis forms, integer overflow-wrap, and the degenerate/error corners. (`np.trapz` was
    ///     REMOVED in NumPy 2.x — only `trapezoid` exists.)
    /// </summary>
    [TestClass]
    public class NpTrapezoidTest
    {
        // ---------------------------- docstring examples ----------------------------

        [TestMethod]
        public void Trapezoid_Basic()
        {
            // np.trapezoid([1,2,3]) -> 4.0 (a 0-d scalar)
            var r = np.trapezoid(np.array(new long[] { 1, 2, 3 }));
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(4.0);
        }

        [TestMethod]
        public void Trapezoid_Dx()
        {
            // np.trapezoid([1,2,3], dx=2) -> 8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), dx: 2).GetDouble(0).Should().Be(8.0);
        }

        [TestMethod]
        public void Trapezoid_XCoords()
        {
            // np.trapezoid([1,2,3], x=[4,6,8]) -> 8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 4, 6, 8 }))
                .GetDouble(0).Should().Be(8.0);
        }

        [TestMethod]
        public void Trapezoid_ReversedX_IntegratesInReverse()
        {
            // np.trapezoid([1,2,3], x=[8,6,4]) -> -8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 8, 6, 4 }))
                .GetDouble(0).Should().Be(-8.0);
        }

        // ---------------------------- dtype tier (NEP50) ----------------------------

        [TestMethod]
        public void Trapezoid_IntegerInputs_ToFloat64()
        {
            foreach (var t in new[] { np.int8, np.uint8, np.int16, np.uint16, np.int32,
                                      np.uint32, np.int64, np.uint64 })
            {
                var r = np.trapezoid(np.array(new double[] { 1, 2, 3, 4 }).astype(t));
                r.dtype.Should().Be(np.float64, $"integer {t} integrates to float64");
                r.GetDouble(0).Should().Be(7.5);
            }
        }

        [TestMethod]
        public void Trapezoid_Float32_StaysFloat32()
        {
            var r = np.trapezoid(np.array(new float[] { 1, 2, 3, 4 }));
            r.dtype.Should().Be(np.float32);
            ((double)r.GetAtIndex<float>(0)).Should().Be(7.5);
        }

        [TestMethod]
        public void Trapezoid_Float16_StaysFloat16()
        {
            var r = np.trapezoid(np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 }));
            r.dtype.Should().Be(np.float16);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(7.5);
        }

        [TestMethod]
        public void Trapezoid_Bool_ToFloat64()
        {
            // bool -> float64; the bool+bool sum is logical, then *1.0/2.0 promotes.
            var r = np.trapezoid(np.array(new bool[] { true, true, true, true, true }));
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(2.0);
        }

        [TestMethod]
        public void Trapezoid_Complex_StaysComplex128()
        {
            var r = np.trapezoid(np.array(new System.Numerics.Complex[]
                { new(1, 1), new(2, 2), new(3, 3) }));
            r.dtype.Should().Be(np.complex128);
            r.GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(4, 4));
        }

        [TestMethod]
        public void Trapezoid_Uint8_Overflow_WrapsBeforePromotion()
        {
            // NumPy computes y[1:]+y[:-1] in uint8 (wraps at 256) BEFORE the float promotion:
            // (200+180)%256=124, (180+150)%256=74, (150+240)%256=134; sum/2 = (124+74+134)/2 = 166.
            var r = np.trapezoid(np.array(new byte[] { 200, 180, 150, 240 }));
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(166.0);
        }

        // ---------------------------- axis handling ----------------------------

        [TestMethod]
        public void Trapezoid_2D_Axis0()
        {
            // np.trapezoid(arange(6).reshape(2,3), axis=0) -> [1.5, 2.5, 3.5]
            var r = np.trapezoid(np.arange(6).reshape(2, 3), axis: 0);
            r.shape.Should().Equal(3);
            r.Data<double>().Should().Equal(new[] { 1.5, 2.5, 3.5 });
        }

        [TestMethod]
        public void Trapezoid_2D_Axis1_IsDefault()
        {
            // axis=1 == default axis=-1 -> [2., 8.]
            np.trapezoid(np.arange(6).reshape(2, 3), axis: 1).Data<double>().Should().Equal(new[] { 2.0, 8.0 });
            np.trapezoid(np.arange(6).reshape(2, 3)).Data<double>().Should().Equal(new[] { 2.0, 8.0 });
        }

        [TestMethod]
        public void Trapezoid_2D_X1D_AlongAxis()
        {
            // np.trapezoid(arange(6).reshape(2,3), x=[0,2,4], axis=1) -> [4., 16.]
            var r = np.trapezoid(np.arange(6).reshape(2, 3), np.array(new long[] { 0, 2, 4 }), axis: 1);
            r.Data<double>().Should().Equal(new[] { 4.0, 16.0 });
        }

        // ---------------------------- degenerate corners ----------------------------

        [TestMethod]
        public void Trapezoid_SingleElement_IsZero()
        {
            var r = np.trapezoid(np.array(new long[] { 5 }));
            r.ndim.Should().Be(0);
            r.GetDouble(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Trapezoid_Empty_IsZero()
        {
            np.trapezoid(np.array(new double[] { })).GetDouble(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Trapezoid_2D_UnitAxis_IsZeros()
        {
            // np.trapezoid(zeros((2,1))+3, axis=1) -> [0., 0.]
            var a = np.full(new Shape(2, 1), 3.0);
            np.trapezoid(a, axis: 1).Data<double>().Should().Equal(new[] { 0.0, 0.0 });
        }

        // ---------------------------- errors ----------------------------

        [TestMethod]
        public void Trapezoid_0D_Throws()
        {
            // NumPy leaks an IndexError; NumSharp raises the clearer axis-out-of-bounds error.
            Action act = () => np.trapezoid(np.array(5.0));
            act.Should().Throw<ArgumentOutOfRangeException>();
        }

        // ---------------------- layout-order (KEEPORDER) parity ----------------------
        // np.sum is LAYOUT-ORDER-DEPENDENT: NumPy allocates trapezoid's reduction input
        // d*(y1+y2)/2 in KEEPORDER (its ufunc output layout, matching the operand's stride
        // permutation), so a non-C-contiguous `y` sums in a different order than a C-order
        // one — a former 1-ULP float16/complex128 divergence. RelayoutHalfToNumpyKeepOrder
        // now re-lays the intermediate into NumPy's exact KEEPORDER before the sum. These
        // pin BIT-for-BIT equality with NumPy 2.4.2 on F-contiguous / transposed / permuted
        // inputs, AND assert the C-order result differs (so the checks are non-vacuous).
        // All expected uint16/uint64 bit patterns were probed directly from NumPy 2.4.2.

        // The (5,7) float16 base whose C-order and F-order trapezoid diverge by one ULP.
        private static readonly ushort[] F16_5x7 =
        {
            14592, 15149, 14901, 13109, 13517, 15101,  7524, 14994, 14944, 14205, 13529, 13428,
            13332, 14111, 14345, 14446, 15351, 14935, 14586, 15337, 13028, 12576, 14566, 10656,
            10385, 14366, 14198, 15190, 14601, 14365, 14323, 13292,  8714, 12840, 14729,
        };

        /// <summary>Reconstructs a float16 NDArray from raw uint16 bit patterns.</summary>
        private static NDArray HalfFromBits(ushort[] bits, params int[] shape)
        {
            var h = new Half[bits.Length];
            for (int i = 0; i < bits.Length; i++) h[i] = BitConverter.UInt16BitsToHalf(bits[i]);
            return np.array(h).reshape(shape);
        }

        /// <summary>Raw uint16 bit patterns of a float16 result, in C-order.</summary>
        private static ushort[] HalfBitsOf(NDArray a)
        {
            var r = new ushort[a.size];
            for (int i = 0; i < r.Length; i++) r[i] = BitConverter.HalfToUInt16Bits(a.GetAtIndex<Half>(i));
            return r;
        }

        [TestMethod]
        public void Trapezoid_Float16_FortranLayout_MatchesNumpyKeepOrder()
        {
            var cbase = HalfFromBits(F16_5x7, 5, 7);
            var f = np.asfortranarray(cbase);

            // F-order (KEEPORDER) results — bit-exact with NumPy 2.4.2.
            HalfBitsOf(np.trapezoid(f, axis: 0)).Should()
                .Equal(new ushort[] { 16442, 16726, 16458, 15736, 15944, 16509, 16309 });
            HalfBitsOf(np.trapezoid(f, axis: 1)).Should()
                .Equal(new ushort[] { 17093, 16759, 17488, 16492, 16447 });

            // The C-order results genuinely differ (idx 3/6 on axis 0, idx 0/1 on axis 1),
            // so the F-order assertions above are non-vacuous.
            HalfBitsOf(np.trapezoid(cbase, axis: 0)).Should()
                .Equal(new ushort[] { 16442, 16726, 16458, 15737, 15944, 16509, 16308 });
            HalfBitsOf(np.trapezoid(cbase, axis: 1)).Should()
                .Equal(new ushort[] { 17094, 16760, 17488, 16492, 16447 });
        }

        [TestMethod]
        public void Trapezoid_Float16_Transposed_MatchesNumpyKeepOrder()
        {
            // Transpose of a C-contiguous array is F-contiguous; NumPy's half preserves it.
            var t = HalfFromBits(F16_5x7, 5, 7).transpose();   // (7,5), F-contiguous
            HalfBitsOf(np.trapezoid(t, axis: 0)).Should()
                .Equal(new ushort[] { 17094, 16760, 17488, 16492, 16447 });
            HalfBitsOf(np.trapezoid(t, axis: 1)).Should()
                .Equal(new ushort[] { 16442, 16726, 16458, 15737, 15944, 16509, 16308 });
        }

        [TestMethod]
        public void Trapezoid_Float16_3D_Permuted_MatchesNumpyKeepOrder()
        {
            // A partial permutation (0,2,1) moves the fastest axis away from last, so NumPy's
            // KEEPORDER half is neither C- nor F-contiguous. axis=1 is the divergent reduction.
            ushort[] baseBits =
            {
                    0, 12434, 13458, 14043, 14482, 14775, 15067, 15360, 15506, 15653, 15799, 15945,
                16091, 16238, 16384, 16457, 16530, 16603, 16677, 16750, 16823, 16896, 16969, 17042,
            };
            var p = HalfFromBits(baseBits, 2, 3, 4).transpose(new[] { 0, 2, 1 });   // (2,4,3)

            HalfBitsOf(np.trapezoid(p, axis: 1)).Should()
                .Equal(new ushort[] { 14628, 16567, 17426, 17865, 18304, 18588 });
            // C-order copy of the same data diverges at index 1 (16566 vs 16567).
            HalfBitsOf(np.trapezoid(np.ascontiguousarray(p), axis: 1)).Should()
                .Equal(new ushort[] { 14628, 16566, 17426, 17865, 18304, 18588 });
        }

        [TestMethod]
        public void Trapezoid_Complex128_FortranLayout_MatchesNumpyKeepOrder()
        {
            ulong[] re =
            {
                4603805578966219945, 4606256603024712909, 4605161975116366175, 4597281964425917712,
                4599078934993407444, 4606043489487626052, 4572720505015638656, 4605572187543305792,
                4605354582709908675, 4602101186634948366, 4599130566815924470, 4598687289849535700,
                4598262942237854670, 4601689401859204016, 4602719786247343358, 4603160680482382897,
                4607141888956321124, 4605314883393280097, 4603779311837014181, 4607082980650445231,
                4596925333675683184, 4594940267138395088, 4603692485812383801, 4586493531777579952,
                4585302891121850208, 4602812925743499171, 4602070041045637328, 4606436332428451530,
                4603842785795791311, 4602805979628576750, 4602622495991859856, 4598085685175422820,
                4577952414617972160, 4596100038082195660, 4604408490748740146,
            };
            ulong[] im =
            {
                4596395639229702204, 4600328594280817784, 4570756998456803584, 4605651624838546151,
                4594733067227427780, 4598492260431178192, 4606104546666537068, 4602767006947997084,
                4605805670612996722, 4603937279534606157, 4604856498269542655, 4591257373829052456,
                4603049409769886418, 4602748825253659095, 4606023546929651011, 4600179574644139972,
                4603563182629623161, 4588699882616220032, 4600654573658062848, 4599490925392448264,
                4594579535841518116, 4605528139505612269, 4600507114465110514, 4606990996760324494,
                4603489392282864961, 4603625081783849029, 4603921781872280383, 4604268141677216325,
                4594600731226044228, 4601603602184925650, 4597799221644350152, 4600922384659339322,
                4591632685005504304, 4606892639645400787, 4596914357110688172,
            };
            var c = new System.Numerics.Complex[re.Length];
            for (int i = 0; i < c.Length; i++)
                c[i] = new System.Numerics.Complex(BitConverter.UInt64BitsToDouble(re[i]),
                                                   BitConverter.UInt64BitsToDouble(im[i]));
            var cbase = np.array(c).reshape(5, 7);
            var f = np.asfortranarray(cbase);

            var rF = np.trapezoid(f, axis: 0);
            var realBits = new ulong[rF.size];
            for (int i = 0; i < realBits.Length; i++)
                realBits[i] = BitConverter.DoubleToUInt64Bits(rF.GetAtIndex<System.Numerics.Complex>(i).Real);
            // F-order (KEEPORDER) real parts — bit-exact with NumPy 2.4.2.
            realBits.Should().Equal(new ulong[]
            {
                4611940808611965020, 4613191959337571042, 4612009504798757630, 4608838553075109488,
                4609756108069491522, 4612233236306052718, 4611353659485408152,
            });

            // C-order diverges in the last bit at indices 0/1/3/6 — the assertion is non-vacuous.
            var rC = np.trapezoid(cbase, axis: 0);
            BitConverter.DoubleToUInt64Bits(rC.GetAtIndex<System.Numerics.Complex>(0).Real)
                .Should().Be(4611940808611965019);
        }
    }
}
