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

        // ------------------- NpyIter layout rule (broadcast y, unit-axis x, F x) -------------------
        // NumPy lays out every ufunc output with NpyIter's rule, where an operand abstains on an axis
        // with extent 1 OR stride 0 (a broadcast axis). The earlier relayout used concatenate's vote
        // (abstaining only on extent 1), let an N-D x vote with the result's dims, and skipped a
        // C-contiguous y entirely — three layouts whose sum ran in the wrong order. Each test pins NumPy
        // 2.4.2's bits and the bits the wrong layout gives, so it fails if the layout regresses.

        /// <summary><c>((arange(n) * 37) % 101) * 0.1</c>: inexact values, so the summation order shows.</summary>
        private static NDArray Inexact(long n) => ((np.arange(n) * 37) % 101) * 0.1;

        /// <summary><c>((arange(n) * 37) % 101) / 64.0</c>: values exact in float16.</summary>
        private static NDArray Sixtyfourths(long n) => ((np.arange(n) * 37) % 101) / 64.0;

        /// <summary>Raw uint64 bit patterns of a float64 result, in C-order.</summary>
        private static ulong[] DoubleBitsOf(NDArray a)
        {
            var r = new ulong[a.size];
            for (int i = 0; i < r.Length; i++) r[i] = BitConverter.DoubleToUInt64Bits(a.GetAtIndex<double>(i));
            return r;
        }

        [TestMethod]
        public void Trapezoid_BroadcastRowY_SumsInNumpysOrder()
        {
            // y = broadcast_to(row, (33, 7)): axis 0 has stride 0, so it casts no vote and the half is C —
            // axis 0 is summed sequentially. The old vote made the stride-0 axis the innermost one (F),
            // summing it pairwise (float16: in float32).
            var y16 = np.broadcast_to(Inexact(7).astype(np.float16), new Shape(33, 7));
            var r16 = np.trapezoid(y16, axis: 0);
            HalfBitsOf(r16).Should().Equal(new ushort[] { 0, 22370, 23394, 20480, 22707, 23602, 21504 });

            // Non-vacuous: the same half summed F-ordered gives NumPy's other answer.
            var h16 = (y16["1:"] + y16[":-1"]) * 1.0 / 2.0;
            HalfBitsOf(np.sum(np.asfortranarray(h16), 0)).Should()
                .Equal(new ushort[] { 0, 22374, 23398, 20480, 22707, 23603, 21504 });

            var y64 = np.broadcast_to(Inexact(7), new Shape(33, 7));
            DoubleBitsOf(np.trapezoid(y64, axis: 0)).Should().Equal(new ulong[]
            {
                0, 4638032076247505310, 4642535675874875806, 4629700416936869888, 4639495746126400717,
                4643436395800349902, 4634204016564240384,
            });
        }

        [TestMethod]
        public void Trapezoid_XWithUnitAxis_AgainstFOrderedY_SumsInNumpysOrder()
        {
            // x (1, 40) has one axis of extent > 1, so it never votes: y's F order lays the half out and
            // axis 1 (outer in memory) is summed sequentially. The old vote let x vote with the RESULT's
            // dims, C won the conflict, and axis 1 was summed pairwise ([18364, 18312, 18260, ...]).
            var y = np.asfortranarray(Sixtyfourths(240).astype(np.float16).reshape(6, 40));
            var x = (np.arange(40) * 0.25).reshape(1, 40).astype(np.float16);
            HalfBitsOf(np.trapezoid(y, x, axis: 1)).Should()
                .Equal(new ushort[] { 18361, 18308, 18256, 18357, 18356, 18304 });
        }

        [TestMethod]
        public void Trapezoid_CContiguousYWithUnitAxis_AgainstFOrderedX_SumsInNumpysOrder()
        {
            // y (1, 40) is C-contiguous but has one axis of extent > 1, so only the F-ordered x's
            // difference votes and the half is F. The old code skipped every C-contiguous y and summed
            // C-ordered ([17340, 49985, 49201, ...]).
            var y = Sixtyfourths(40).astype(np.float16).reshape(1, 40);
            var x = np.asfortranarray(((np.arange(240) * 0.125) % 7).astype(np.float16).reshape(6, 40));
            HalfBitsOf(np.trapezoid(y, x, axis: 1)).Should()
                .Equal(new ushort[] { 17337, 49989, 49203, 17337, 50278, 49595 });
        }

        [TestMethod]
        public void Trapezoid_FlatRoute_Float16LongRows()
        {
            // A dense y integrated along its innermost memory axis runs as one contiguous pass; the
            // 299-sample rows are summed pairwise in float32 like NumPy's (sequential gives
            // [22344, 22344, 22347]).
            var y = Sixtyfourths(900).astype(np.float16).reshape(3, 300);
            HalfBitsOf(np.trapezoid(y, dx: 0.5, axis: 1)).Should().Equal(new ushort[] { 22348, 22346, 22351 });
            var yf = np.asfortranarray(y.T.copy());     // (300, 3) F: axis 0 is innermost
            HalfBitsOf(np.trapezoid(yf, dx: 0.5, axis: 0)).Should().Equal(new ushort[] { 22348, 22346, 22351 });
        }

        [TestMethod]
        public void Trapezoid_ResultLayout_FollowsNumpysReduction()
        {
            // NumPy's reduction lays its result out along the half's remaining axes in memory order.
            var f = np.asfortranarray(Inexact(120).reshape(4, 5, 6));
            var rf = np.trapezoid(f, axis: 0);
            rf.Shape.IsFContiguous.Should().BeTrue("an F half (4,5,6) reduced over axis 0 is (5,6) F in NumPy");
            rf.Shape.IsContiguous.Should().BeFalse();

            var p = Inexact(120).reshape(5, 6, 4).transpose(new[] { 2, 0, 1 });   // (4,5,6), strides (1,24,4)
            var rp = np.trapezoid(p, axis: 1);
            rp.shape.Should().Equal(4, 6);
            rp.Shape.IsFContiguous.Should().BeTrue("NumPy: strides (8, 32) bytes");
            rp.Shape.strides.Should().Equal(1, 4);
        }

        // ------------------- the axis, read the way NumPy reads it -------------------

        [TestMethod]
        public void Trapezoid_XOfHigherRank_DiffsAndSumsItsOwnLastAxis()
        {
            // NumPy passes the RAW axis to diff(x, axis) and .sum(axis): a higher-rank x is differenced
            // along its own last axis and the product's last axis is summed.
            var x = ((np.arange(48) * 0.375) % 5.5).reshape(3, 2, 8);
            var r = np.trapezoid(Inexact(8), x);
            r.shape.Should().Equal(3, 2);
            DoubleBitsOf(r).Should().Equal(new ulong[]
            {
                4622533360342374811, 13845138377885614078, 4622533360342374811, 13848943567727021260,
                4622533360342374811, 13840019051746689024,
            });

            var x2 = ((np.arange(24) * 0.625) % 4.25).reshape(4, 2, 3);
            var r2 = np.trapezoid(Inexact(6).reshape(2, 3), x2);
            r2.shape.Should().Equal(4, 2);
            DoubleBitsOf(r2).Should().Equal(new ulong[]
            {
                4616893305496535040, 4618300680380088320, 13837844657551599207, 4618300680380088320,
                13849402723782780519, 4618300680380088320, 4616893305496535040, 4618300680380088320,
            });

            // axis 0: x's axis 0 is differenced, y's axis 0 sliced, the product's axis 0 summed.
            var r3 = np.trapezoid(Inexact(6).reshape(2, 3), x2, axis: 0);
            r3.shape.Should().Equal(2, 3);
            DoubleBitsOf(r3).Should().Equal(new ulong[]
            {
                4608871268660281344, 4622691690016774554, 4626808261551167898, 13828302655841107968,
                13842151224695272244, 13846232611857576756,
            });
        }

        [TestMethod]
        public void Trapezoid_XErrorsComeBeforeYAxisErrors()
        {
            // NumPy resolves the spacing before slicing y: a 0-D x fails in diff even when y is 0-D, and an
            // axis outside an N-D x is reported against x's rank.
            Action zeroD = () => np.trapezoid(np.array(5.0), np.array(2.0));
            zeroD.Should().Throw<ArgumentException>().WithMessage("diff requires input that is at least one dimensional*");

            Action outsideX = () => np.trapezoid(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 2, 3)), axis: 5);
            outsideX.Should().Throw<ArgumentOutOfRangeException>().WithMessage("axis 5 is out of bounds for array of dimension 3*");

            // An axis inside x but outside y fails at y's slicing (NumPy: IndexError).
            Action outsideY = () => np.trapezoid(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 2, 3)), axis: 2);
            outsideY.Should().Throw<ArgumentOutOfRangeException>().WithMessage("axis 2 is out of bounds for array of dimension 2*");

            // A spacing that does not broadcast: NumPy's multiply error, verbatim.
            Action mismatch = () => np.trapezoid(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 5)), axis: 1);
            mismatch.Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with shapes (2,4) (2,2) ");
        }

        // ------------------- the N-D difference, computed in the fused pass -------------------

        [TestMethod]
        public void Trapezoid_BoolX_DiffsByNotEqual()
        {
            // NumPy's diff of a bool array is not_equal, so d is 0/1.
            var y = Inexact(10).reshape(2, 5);
            var xb = np.array(new bool[,] { { true, false, false, true, true }, { false, false, true, true, false } });
            DoubleBitsOf(np.trapezoid(y, xb, axis: 1)).Should().Equal(new ulong[] { 4618497712863785780, 4621847265086642586 });

            // bool y with bool x: or-sum times and-product, then the float division.
            var yb = np.array(new bool[,] { { true, false, true, true, false }, { true, true, false, false, true } });
            var r = np.trapezoid(yb, xb, axis: 1);
            r.dtype.Should().Be(np.float64);
            r.Data<double>().Should().Equal(new[] { 1.0, 1.0 });
        }

        [TestMethod]
        public void Trapezoid_Uint8X_DiffWrapsAtUint8()
        {
            // The difference wraps in x's own dtype before it meets y (5 - 10 → 251), as NumPy's diff does.
            var y = Inexact(10).reshape(2, 5);
            var xu = np.array(new byte[,] { { 10, 5, 7, 7, 250 }, { 0, 255, 1, 200, 3 } });
            var r = np.trapezoid(y, xu, axis: 1);
            DoubleBitsOf(r).Should().Equal(new ulong[] { 4652851733771321344, 4659259577586837095 });
            r.GetDouble(0).Should().Be(1168.0);
        }
    }
}
