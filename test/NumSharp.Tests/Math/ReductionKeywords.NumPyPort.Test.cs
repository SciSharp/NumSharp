using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.sum / np.prod / np.mean / np.var / np.std and their ndarray methods with NumPy's whole signature (np-layer dtype
    ///     plan, P10 item 1, decision D10): one overload each, and the keywords the overloads lacked — <c>initial=</c> and
    ///     <c>where=</c> for sum / prod, <c>where=</c> for mean, <c>where=</c> / <c>mean=</c> / <c>correction=</c> for var /
    ///     std. Every expected value is NumPy 2.4.2's for the same call (probed 2026-11-02; floats as exact round-trip
    ///     values, NaN payloads and the larger cases as raw bits).
    /// </summary>
    /// <remarks>
    ///     The keyword paths are NumPy's statements: <c>PyUFunc_ReduceWrapper</c> through NumPy's reduction iterator for
    ///     sum / prod (<c>Default.Reduction.AddMul.Keywords.cs</c>), <c>_methods._mean</c> / <c>_var</c> / <c>_std</c> for
    ///     the statistics (<c>Default.Reduction.MeanVarStd.Keywords.cs</c>).
    /// </remarks>
    [TestClass]
    public class ReductionKeywordsNumPyPortTests
    {
        /// <summary>The (2, 3) float64 operand most cases reduce.</summary>
        /// <returns>A fresh C-contiguous array <c>[[1.5, -2, 3.25], [4, 0.5, -6]]</c>.</returns>
        private static NDArray F() => np.array(new[,] { { 1.5, -2.0, 3.25 }, { 4.0, 0.5, -6.0 } });

        /// <summary>The (2, 3) mask most cases pass.</summary>
        /// <returns>A fresh bool array <c>[[T, F, T], [F, T, T]]</c>.</returns>
        private static NDArray M2() => np.array(new[,] { { true, false, true }, { false, true, true } });

        /// <summary>The (3, 3) int32 operand of NumPy's own np.mean docstring example.</summary>
        /// <returns>A fresh int32 array <c>[[5, 9, 13], [14, 10, 12], [11, 15, 19]]</c>.</returns>
        private static NDArray A() => np.array(new[,] { { 5, 9, 13 }, { 14, 10, 12 }, { 11, 15, 19 } });

        /// <summary>The (3, 1) row mask selecting rows 0 and 2.</summary>
        /// <returns>A fresh bool array <c>[[T], [F], [T]]</c>.</returns>
        private static NDArray W3() => np.array(new[,] { { true }, { false }, { true } });

        /// <summary>
        ///     Asserts a result's dtype, shape and values exactly (a float compared bit for bit through its double value, NaN
        ///     equal to NaN; an integer or bool by value).
        /// </summary>
        /// <param name="r">The result.</param>
        /// <param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="values">NumPy's values in C order.</param>
        private static void Expect(NDArray r, string dtype, long[] shape, params double[] values)
        {
            r.dtype.name.Should().Be(dtype);
            r.shape.Should().Equal(shape);
            var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            flat.size.Should().Be(values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                double got = flat.typecode switch
                {
                    NPTypeCode.Half => (double)flat.GetAtIndex<Half>(i),
                    NPTypeCode.Single => flat.GetAtIndex<float>(i),
                    NPTypeCode.Boolean => flat.GetAtIndex<bool>(i) ? 1 : 0,
                    _ => Convert.ToDouble(flat.GetAtIndex(i)),
                };
                if (double.IsNaN(values[i]))
                    double.IsNaN(got).Should().BeTrue($"element {i} is NaN in NumPy");
                else
                    BitConverter.DoubleToInt64Bits(got).Should().Be(BitConverter.DoubleToInt64Bits(values[i]), $"element {i} is {values[i]:R} in NumPy, got {got:R}");
            }
        }

        /// <summary>The raw bits of a 0-d (or one-element) float64 result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>The bits.</returns>
        private static ulong Bits(NDArray r) => BitConverter.DoubleToUInt64Bits(r.GetAtIndex<double>(0));

        /// <summary>A float64 from its raw bits (a NaN with a payload).</summary>
        /// <param name="bits">The bits.</param>
        /// <returns>The value.</returns>
        private static double FromBits(ulong bits) => BitConverter.UInt64BitsToDouble(bits);

        /// <summary>A float64 from Python's <c>float.hex()</c> text.</summary>
        /// <param name="hex">The text (<c>-0x1.b25af8592fe55p+6</c>).</param>
        /// <returns>The value.</returns>
        private static double Hex(string hex)
        {
            bool neg = hex.StartsWith("-");
            string body = neg ? hex[1..] : hex;
            int p = body.IndexOf('p');
            string mant = body[2..p];
            int exp = int.Parse(body[(p + 1)..]);
            string frac = mant.Substring(2);
            ulong bits = ((ulong)(exp + 1023) << 52) | Convert.ToUInt64(frac.PadRight(13, '0'), 16);
            double v = BitConverter.UInt64BitsToDouble(bits);
            return neg ? -v : v;
        }

        // ------------------------------------------------------------------------------------------------ signatures

        /// <summary>
        ///     Each function has ONE public overload with NumPy's parameter names, order and defaults (D10, D19: the old
        ///     overloads are removed, not obsoleted), and each ndarray method likewise (ndarray.var / std without
        ///     <c>correction</c>, which NumPy 2.4.2's methods refuse).
        /// </summary>
        [TestMethod]
        public void OneOverloadEach_NumPysSignature()
        {
            var np5 = new (string Name, string[] Params)[]
            {
                ("sum", new[] { "a", "axis", "dtype", "out", "keepdims", "initial", "where" }),
                ("prod", new[] { "a", "axis", "dtype", "out", "keepdims", "initial", "where" }),
                ("mean", new[] { "a", "axis", "dtype", "out", "keepdims", "where" }),
                ("var", new[] { "a", "axis", "dtype", "out", "ddof", "keepdims", "where", "mean", "correction" }),
                ("std", new[] { "a", "axis", "dtype", "out", "ddof", "keepdims", "where", "mean", "correction" }),
            };
            foreach (var (name, ps) in np5)
            {
                var ms = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).ToArray();
                ms.Should().HaveCount(1, $"np.{name} keeps one overload");
                ms[0].GetParameters().Select(p => p.Name).Should().Equal(ps, $"np.{name} takes NumPy's parameters in NumPy's order");
                ms[0].GetParameters().Skip(1).Should().OnlyContain(p => p.HasDefaultValue, $"every np.{name} parameter after a is optional");
            }

            var nd5 = new (string Name, string[] Params)[]
            {
                ("sum", new[] { "axis", "dtype", "out", "keepdims", "initial", "where" }),
                ("prod", new[] { "axis", "dtype", "out", "keepdims", "initial", "where" }),
                ("mean", new[] { "axis", "dtype", "out", "keepdims", "where" }),
                ("var", new[] { "axis", "dtype", "out", "ddof", "keepdims", "where", "mean" }),
                ("std", new[] { "axis", "dtype", "out", "ddof", "keepdims", "where", "mean" }),
            };
            foreach (var (name, ps) in nd5)
            {
                var ms = typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                    .Where(m => m.Name == name).ToArray();
                ms.Should().HaveCount(1, $"ndarray.{name} keeps one overload");
                ms[0].GetParameters().Select(p => p.Name).Should().Equal(ps, $"ndarray.{name} takes NumPy's parameters in NumPy's order");
            }

            // NumPy's defaults: ddof 0, keepdims false, everything else None.
            var var = typeof(np).GetMethod("var")!.GetParameters();
            var.Single(p => p.Name == "ddof").DefaultValue.Should().Be(0);
            var.Single(p => p.Name == "keepdims").DefaultValue.Should().Be(false);
            var.Where(p => p.Name is "axis" or "dtype" or "out" or "where" or "mean" or "correction").Should().OnlyContain(p => p.DefaultValue == null);
        }

        /// <summary>
        ///     The positional order is NumPy's: the third argument is dtype and the fourth out, so a positional bool in the out
        ///     slot is a C# scalar passed as out (NumPy's <c>TypeError: return arrays must be of ArrayType</c>).
        /// </summary>
        [TestMethod]
        public void PositionalOrder_IsNumPys()
        {
            Expect(np.sum(F(), 0, np.float32), "float32", new long[] { 3 }, 5.5, -1.5, -2.75);
            Expect(np.mean(F(), null, np.float32), "float32", new long[0], 0.2083333283662796);
            new Action(() => np.sum(F(), 0, null, true)).Should().Throw<TypeError>().WithMessage("return arrays must be of ArrayType");
            new Action(() => np.prod(F(), 0, null, true)).Should().Throw<TypeError>().WithMessage("return arrays must be of ArrayType");
            new Action(() => np.mean(F(), 0, null, true)).Should().Throw<TypeError>().WithMessage("return arrays must be of ArrayType");
        }

        // ------------------------------------------------------------------------------------------------ sum / prod

        /// <summary>
        ///     <c>initial=</c> packed into the loop as NumPy's element setter packs a Python value: the NEP 50 accumulator
        ///     loop (int32 / int8 → int64), a truncated float, the range check of a <c>dtype=</c> loop, the TypeError of a
        ///     complex into a real loop, a float32 loop rounding a Python float once.
        /// </summary>
        [TestMethod]
        public void Sum_Initial_PackedIntoTheLoop()
        {
            Expect(np.sum(np.array(new[] { 1, 2, 3 }), initial: 10), "int64", new long[0], 16);
            Expect(np.sum(np.array(new sbyte[] { 100, 100 }), initial: 5), "int64", new long[0], 205);
            Expect(np.sum(np.array(new long[] { 1, 2 }), initial: 2.5), "int64", new long[0], 5);
            Expect(np.sum(np.array(new[] { 1.0f }), initial: 0.1), "float32", new long[0], 1.100000023841858);
            Expect(np.sum(np.array(new[] { 1.0f }), initial: 1e40), "float32", new long[0], double.PositiveInfinity);
            new Action(() => np.sum(np.array(new sbyte[] { 1 }), dtype: np.int8, initial: 1000))
                .Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            new Action(() => np.sum(np.array(new long[] { 1, 2 }), initial: double.NaN))
                .Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            new Action(() => np.sum(np.array(new[] { 1.0, 2.0 }), initial: new Complex(0, 1)))
                .Should().Throw<TypeError>().WithMessage("float() argument must be a string or a real number, not 'complex'");
            new Action(() => np.sum(np.array(new byte[] { 1 }), initial: -1))
                .Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint64");
            new Action(() => np.sum(np.array(new byte[] { 1 }), dtype: np.uint8, initial: -1))
                .Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            // An empty sum is the initial value.
            Expect(np.sum(np.array(new double[0]), initial: 5), "float64", new long[0], 5);
            Expect(np.sum(np.array(new int[0]), initial: 5), "int64", new long[0], 5);
        }

        /// <summary>
        ///     <c>where=</c>: only the selected elements are added, the mask broadcasts against the input, an output with none
        ///     selected keeps the identity (or the initial value), and the C# literals are Python's bools.
        /// </summary>
        [TestMethod]
        public void Sum_Where_SelectsElements()
        {
            Expect(np.sum(np.array(new[] { 1.0, 2.0, 3.0 }), where: np.array(new[] { true, false, true })), "float64", new long[0], 4);
            Expect(np.sum(F(), 0, where: M2()), "float64", new long[] { 3 }, 1.5, 0.5, -2.75);
            Expect(np.sum(F(), 1, where: M2()), "float64", new long[] { 2 }, 4.75, -5.5);
            Expect(np.sum(F(), where: np.array(new[,] { { true }, { false } })), "float64", new long[0], 2.75);
            Expect(np.sum(F(), 1, where: np.array(new[] { true, false, true })), "float64", new long[] { 2 }, 4.75, -2.0);
            Expect(np.sum(F(), where: false), "float64", new long[0], 0);
            Expect(np.sum(F(), where: false, initial: 5), "float64", new long[0], 5);
            Expect(np.sum(F(), where: true), "float64", new long[0], 1.25);
            Expect(np.sum(F(), where: np.array(true)), "float64", new long[0], 1.25);
            Expect(np.sum(np.array(new[] { true, true, false }), where: np.array(new[] { true, false, true })), "int64", new long[0], 1);
            Expect(np.sum(np.array(new[] { true, true, false }), dtype: np.@bool, where: np.array(new[] { true, false, true })), "bool", new long[0], 1);
            Expect(np.sum(np.array(new ushort[] { 1, 2, 3 }), where: np.array(new[] { true, true, false })), "uint64", new long[0], 3);
            Expect(np.sum(np.array(new[] { (Half)1.5, (Half)2.25, (Half)1000 }), where: np.array(new[] { true, true, false })), "float16", new long[0], 3.75);
            var c = np.sum(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(0, 5) }), where: np.array(new[] { true, false, true }), initial: new Complex(0, 1));
            c.dtype.name.Should().Be("complex128");
            c.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 8));
            // keepdims with a mask, along an axis and over every axis; a 0-d input (axis 0 is every axis).
            Expect(np.sum(F(), 1, keepdims: true, where: M2(), initial: 1), "float64", new long[] { 2, 1 }, 5.75, -4.5);
            Expect(np.sum(F(), keepdims: true, where: M2()), "float64", new long[] { 1, 1 }, -0.75);
            Expect(np.sum(np.array(5.0), where: false, initial: 1), "float64", new long[0], 1);
            Expect(np.sum(np.array(5.0), 0, where: true, initial: 1), "float64", new long[0], 6);
        }

        /// <summary>
        ///     The reduce prologue's errors with a mask, NumPy's texts: a non-bool mask, a mask with more axes than the input,
        ///     a mask that does not broadcast, a read-only out (first).
        /// </summary>
        [TestMethod]
        public void Sum_Where_PrologueErrors()
        {
            new Action(() => np.sum(F(), where: np.array(new[,] { { 1L, 0L, 1L }, { 0L, 1L, 1L } })))
                .Should().Throw<ArgumentException>().WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            new Action(() => np.sum(np.arange(3.0), where: np.ones(new Shape(2, 3), np.@bool)))
                .Should().Throw<ValueError>().WithMessage("input operand has more dimensions than allowed by the axis remapping");
            new Action(() => np.sum(np.arange(3.0), where: np.ones(new Shape(2), np.@bool)))
                .Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (3,) (2,) ");
            new Action(() => np.sum(F(), 0, @out: np.broadcast_to(np.zeros(new Shape(3)), new Shape(3)), initial: 1))
                .Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        /// <summary>
        ///     The start value is SET (raw_array_assign_scalar) and the loop adds to it: the identity +0.0 turns a lone -0.0
        ///     into +0.0, an initial -0.0 keeps it; multiply's identity 1 keeps the sign.
        /// </summary>
        [TestMethod]
        public void SumProd_StartValue_SignedZero()
        {
            Bits(np.sum(np.array(new[] { -0.0 }), where: np.array(new[] { true }))).Should().Be(0UL);
            Bits(np.sum(np.array(new[] { -0.0 }), initial: -0.0)).Should().Be(0x8000000000000000UL);
            Bits(np.sum(np.array(new[,] { { -0.0, -0.0 } }), 1, initial: -0.0)).Should().Be(0x8000000000000000UL);
            Bits(np.prod(np.array(new[] { -0.0 }), where: np.array(new[] { true }))).Should().Be(0x8000000000000000UL);
        }

        /// <summary>
        ///     <c>out=</c> with the keywords: the loop is promote_types(out, a) (int16 here, not the int64 accumulator), the
        ///     initial value is cast into the out, and an output the mask skips keeps that value — the out's old contents are
        ///     overwritten by the start value first.
        /// </summary>
        [TestMethod]
        public void Sum_OutWithKeywords()
        {
            var o8 = np.zeros(new Shape(2), np.int8);
            var r = np.sum(np.array(new short[,] { { 1, 2 }, { 3, 4 } }), 0, @out: o8, initial: 100,
                where: np.array(new[,] { { true, false }, { true, false } }));
            ReferenceEquals(r, o8).Should().BeTrue();
            Expect(o8, "int8", new long[] { 2 }, 104, 100);
            var o = np.full(new Shape(3), 7.0);
            np.sum(F(), 0, @out: o, where: M2());
            Expect(o, "float64", new long[] { 3 }, 1.5, 0.5, -2.75);
            var o0 = np.zeros(new Shape(), np.float64);
            np.sum(np.array(new long[] { 1, 2, 3 }), dtype: np.float32, @out: o0, initial: 0.5);
            Expect(o0, "float64", new long[0], 6.5);
        }

        /// <summary>prod with <c>initial=</c> / <c>where=</c>: the same statements with multiply's identity 1.</summary>
        [TestMethod]
        public void Prod_InitialAndWhere()
        {
            Expect(np.prod(np.array(new long[] { 2, 3, 4 }), initial: 2, where: np.array(new[] { true, false, true })), "int64", new long[0], 16);
            Expect(np.prod(np.array(new double[0]), initial: 3), "float64", new long[0], 3);
            Expect(np.prod(np.array(new[] { true, false, true }), where: np.array(new[] { true, false, true })), "int64", new long[0], 1);
            Expect(np.prod(np.array(new[] { true, false, true }), dtype: np.@bool, where: np.array(new[] { true, false, true })), "bool", new long[0], 1);
            Expect(np.prod(F(), 1, where: M2()), "float64", new long[] { 2 }, 4.875, -3.0);
            Expect(np.prod(F(), 0, where: false), "float64", new long[] { 3 }, 1, 1, 1);
            Expect(np.prod(np.array(new sbyte[] { 1 }), initial: 1000), "int64", new long[0], 1000);
            new Action(() => np.prod(np.array(new sbyte[] { 1 }), dtype: np.int8, initial: 1000))
                .Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            Expect(np.prod(np.array(new[] { (Half)1.5, (Half)2.25, (Half)3 }), where: np.array(new[] { true, true, false }), initial: 2), "float16", new long[0], 6.75);
            var c = np.prod(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(0, 5) }), where: np.array(new[] { true, true, false }));
            c.GetAtIndex<Complex>(0).Should().Be(new Complex(5, 5));
        }

        /// <summary>
        ///     Which NaN survives, NumPy's win-amd64 loops: multiply's reduce form keeps the RUNNING value's NaN (the first NaN
        ///     met), its elementwise form the second operand's; add's reduce form keeps the RUN's pairwise NaN over the out's,
        ///     its elementwise form the second operand's.
        /// </summary>
        [TestMethod]
        public void SumProd_NaNPayloads_AreNumPys()
        {
            const ulong p1 = 0x7FF8000000000111, p2 = 0x7FF8000000000222;
            var pair = np.array(new[] { FromBits(p1), FromBits(p2) });
            var all = np.array(new[] { true, true });
            Bits(np.prod(pair, where: all)).Should().Be(p1);
            Bits(np.prod(np.array(new[] { FromBits(p1), 1.0, FromBits(p2) }), where: np.array(new[] { true, false, true }))).Should().Be(p1);
            Bits(np.prod(np.array(new[] { FromBits(p2) }), initial: FromBits(p1))).Should().Be(p1);
            var rows = np.array(new[,] { { FromBits(p1), FromBits(p1), FromBits(p1) }, { FromBits(p2), FromBits(p2), FromBits(p2) } });
            var rowsAll = np.ones(new Shape(2, 3), np.@bool);
            foreach (var v in np.prod(rows, 0, where: rowsAll).ToArray<double>())
                BitConverter.DoubleToUInt64Bits(v).Should().Be(p2);
            Bits(np.sum(pair, where: all)).Should().Be(p1);
            Bits(np.sum(np.array(new[] { FromBits(p2) }), initial: FromBits(p1))).Should().Be(p2);
            foreach (var v in np.sum(rows, 0, where: rowsAll).ToArray<double>())
                BitConverter.DoubleToUInt64Bits(v).Should().Be(p2);
        }

        /// <summary>
        ///     Larger masked reductions, bit for bit: NumPy's iterator cuts every call into the mask's runs, each a pairwise
        ///     sum (add) or a sequential product (multiply), so the bits depend on the runs — C and F operands alike.
        /// </summary>
        [TestMethod]
        public void SumProdMeanVar_LargeMasked_BitForBit()
        {
            var rs = np.random.RandomState(7);
            var big = rs.standard_normal(5000);
            var mb = rs.random_sample(5000) > 0.3;
            np.sum(big, where: mb).GetAtIndex<double>(0).Should().Be(Hex("-0x1.b25af8592fe55p+6"));
            np.sum(big, where: mb, initial: 0.125).GetAtIndex<double>(0).Should().Be(Hex("-0x1.b1daf8592fe55p+6"));
            np.prod(1.0 + big / 100, where: mb).GetAtIndex<double>(0).Should().Be(Hex("0x1.23b4715a4c197p-2"));
            np.mean(big, where: mb).GetAtIndex<double>(0).Should().Be(Hex("-0x1.036c291ecbc09p-5"));
            np.var(big, where: mb).GetAtIndex<double>(0).Should().Be(Hex("0x1.fa0970d892aeap-1"));

            var big2 = big.reshape(50, 100);
            var mb2 = mb.reshape(50, 100);
            string[] axis0 = { "0x1.d3a1a7bb0c839p+0", "-0x1.88fa797a7bc19p+3", "-0x1.5dce0bf165b6dp+3", "-0x1.026b0f8f5ecf5p+3", "0x1.8d85b02863889p+0" };
            string[] axis1 = { "0x1.f8c5baf5b76dbp+2", "0x1.b510c710388e8p+1", "-0x1.03af952154662p+2", "-0x1.39bae513d53e5p+2", "-0x1.92f50e9535717p+2" };
            string[] var1 = { "0x1.e2f702116de60p-1", "0x1.d5d8d447c89b6p-1", "0x1.dea577b2053c5p-1", "0x1.268ff3d6c8d50p+0", "0x1.52d64ba85eebep-1" };
            string[] mean0 = { "0x1.c575f7fb32ec3p-5", "-0x1.e3aa6e209850bp-2", "-0x1.269290cb482e3p-2", "-0x1.a81226a9a2264p-3", "0x1.8179e11f68466p-5" };
            string[] prod0 = { "0x1.0461bafde6145p+0", "0x1.c42fcc8915ae0p-1", "0x1.c9e6ae02f87c6p-1", "0x1.d6e4fc419929fp-1", "0x1.03bd179d8c45cp+0" };
            var s0 = np.sum(big2, 0, where: mb2);
            var s1 = np.sum(big2, 1, where: mb2);
            var s1F = np.sum(np.asfortranarray(big2), 1, where: mb2);
            var v1 = np.var(big2, 1, where: mb2);
            var m0 = np.mean(big2, 0, where: mb2);
            var p0 = np.prod(1.0 + big2 / 100, 0, where: mb2);
            for (int i = 0; i < 5; i++)
            {
                s0.GetAtIndex<double>(i).Should().Be(Hex(axis0[i]), $"sum axis 0 [{i}]");
                s1.GetAtIndex<double>(i).Should().Be(Hex(axis1[i]), $"sum axis 1 [{i}]");
                s1F.GetAtIndex<double>(i).Should().Be(Hex(axis1[i]), $"sum axis 1 of the F copy [{i}]");
                v1.GetAtIndex<double>(i).Should().Be(Hex(var1[i]), $"var axis 1 [{i}]");
                m0.GetAtIndex<double>(i).Should().Be(Hex(mean0[i]), $"mean axis 0 [{i}]");
                p0.GetAtIndex<double>(i).Should().Be(Hex(prod0[i]), $"prod axis 0 [{i}]");
            }

            var f32 = big.astype(np.float32);
            ((double)np.sum(f32, where: mb, initial: 0.5f).GetAtIndex<float>(0)).Should().Be(Hex("-0x1.b05b060000000p+6"));
            np.sum(f32, where: mb, dtype: np.float64).GetAtIndex<double>(0).Should().Be(Hex("-0x1.b25af835c1200p+6"));
        }

        /// <summary>
        ///     Text: StringDType's add concatenates from the initial value over the selected elements, and has no identity (a
        ///     mask needs <c>initial=</c>); a fixed-width input has no add loop.
        /// </summary>
        [TestMethod]
        public void Sum_StringDType_InitialAndWhere()
        {
            var t = np.array(new[] { "a", "b" }, np.dtypes.StringDType());
            np.sum(t, initial: "x").GetAtIndex(0).ToString().Should().Be("xab");
            var t3 = np.array(new[] { "a", "b", "c" }, np.dtypes.StringDType());
            np.sum(t3, where: np.array(new[] { true, false, true }), initial: "z").GetAtIndex(0).ToString().Should().Be("zac");
            new Action(() => np.sum(t3, where: np.array(new[] { true, false, true })))
                .Should().Throw<ValueError>().WithMessage("reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
            new Action(() => np.sum(np.array(new[] { "a", "b" }), initial: "x"))
                .Should().Throw<TypeError>().WithMessage("the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U1'), dtype('<U1'), dtype('<U2'))");
        }

        // ------------------------------------------------------------------------------------------------ mean

        /// <summary>
        ///     <c>np.mean</c> with <c>where=</c>: each output divides its masked sum by its own count (NumPy's docstring
        ///     example first), an output with none selected is 0 / 0, and the dtype rules are _mean's (float16 sums in
        ///     float32 and returns float16; an int32 <c>dtype=</c> casts the NaN to INT_MIN).
        /// </summary>
        [TestMethod]
        public void Mean_Where_PerOutputCounts()
        {
            Expect(np.mean(A(), where: np.array(new[,] { { true }, { false }, { false } })), "float64", new long[0], 9);
            Expect(np.mean(F(), 0, where: M2()), "float64", new long[] { 3 }, 1.5, 0.5, -1.375);
            Expect(np.mean(F(), 1, keepdims: true, where: M2()), "float64", new long[] { 2, 1 }, 2.375, -2.75);
            Expect(np.mean(F(), where: false), "float64", new long[0], double.NaN);
            Expect(np.mean(F(), 0, where: np.array(new[,] { { true, false, true }, { true, false, false } })), "float64", new long[] { 3 }, 2.75, double.NaN, 3.25);
            Expect(np.mean(F().astype(np.float32), where: M2()), "float32", new long[0], -0.1875);
            Expect(np.mean(F().astype(np.float32), 1, where: M2()), "float32", new long[] { 2 }, 2.375, -2.75);
            Expect(np.mean(F().astype(np.float16), where: M2()), "float16", new long[0], -0.1875);
            Expect(np.mean(F().astype(np.float16), 1, where: M2()), "float16", new long[] { 2 }, 2.375, -2.75);
            Expect(np.mean(A(), where: W3(), dtype: np.float32), "float32", new long[0], 12);
            Expect(np.mean(A(), 1, where: W3(), dtype: np.int32), "int32", new long[] { 3 }, 9, int.MinValue, 15);
            Expect(np.mean(np.array(5.0), where: false), "float64", new long[0], double.NaN);
            Expect(np.mean(np.array(new[] { true, false, true }), where: np.array(new[] { true, true, false })), "float64", new long[0], 0.5);
            var c = np.mean(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(0, 5) }), where: np.array(new[] { true, false, true }));
            c.GetAtIndex<Complex>(0).Should().Be(new Complex(0.5, 3.5));
            // where=True is no mask; an explicit 0-d true ARRAY is a mask (the same values; ufunc.reduce's axis rule, so a
            // 0-d input takes axis 0 with it and refuses it without).
            Expect(np.mean(F(), where: true), "float64", new long[0], 0.20833333333333334);
            Expect(np.mean(F(), where: np.array(true)), "float64", new long[0], 0.20833333333333334);
            Expect(np.mean(np.array(5.0), 0, where: np.array(true)), "float64", new long[0], 5);
            new Action(() => np.mean(np.array(5.0), 0, where: true)).Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        /// <summary>
        ///     <c>np.mean</c>'s out with a mask: NumPy stores the masked sum into out and divides it there (an int32 out holds
        ///     the truncated means); errors in NumPy's order (the count's axis error first, then the sum's prologue).
        /// </summary>
        [TestMethod]
        public void Mean_Where_OutAndErrors()
        {
            var oi = np.zeros(new Shape(3), np.int32);
            ReferenceEquals(np.mean(A(), 0, @out: oi, where: W3()), oi).Should().BeTrue();
            Expect(oi, "int32", new long[] { 3 }, 8, 12, 16);
            var of = np.zeros(new Shape(3), np.float32);
            np.mean(A(), 0, @out: of, where: W3());
            Expect(of, "float32", new long[] { 3 }, 8, 12, 16);
            new Action(() => np.mean(F(), where: np.array(new[,] { { 1L, 0L, 1L }, { 0L, 1L, 1L } })))
                .Should().Throw<ArgumentException>().WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
            new Action(() => np.mean(F(), 5, where: M2())).Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            new Action(() => np.mean(F(), 9, where: np.ones(new Shape(2, 3), np.int64)))
                .Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 2*");
            new Action(() => np.mean(F(), 0, @out: np.broadcast_to(np.zeros(new Shape(3)), new Shape(3)), where: M2()))
                .Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        // ------------------------------------------------------------------------------------------------ var / std

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> with <c>where=</c>: per-output counts, means and divisors (ddof per output, a
        ///     count at or below ddof is NaN), the dtype rules of _var, std's square root, out= into the last sum.
        /// </summary>
        [TestMethod]
        public void VarStd_Where()
        {
            Expect(np.var(F(), where: M2()), "float64", new long[0], 12.23046875);
            Expect(np.var(F(), 0, where: M2()), "float64", new long[] { 3 }, 0, 0, 21.390625);
            Expect(np.var(F(), 1, ddof: 1, where: M2()), "float64", new long[] { 2 }, 1.53125, 21.125);
            Expect(np.var(F(), 0, ddof: 1, where: M2()), "float64", new long[] { 3 }, double.NaN, double.NaN, 42.78125);
            Expect(np.var(F(), where: false), "float64", new long[0], double.NaN);
            Expect(np.std(F(), where: M2()), "float64", new long[0], 3.4972087083844454);
            Expect(np.std(F(), 1, keepdims: true, where: M2()), "float64", new long[] { 2, 1 }, 0.875, 3.25);
            Expect(np.var(A(), where: W3()), "float64", new long[0], 19.666666666666668);
            Expect(np.var(F().astype(np.float32), 1, where: M2()), "float32", new long[] { 2 }, 0.765625, 10.5625);
            Expect(np.var(F().astype(np.float16), 1, where: M2()), "float16", new long[] { 2 }, 0.765625, 10.5625);
            Expect(np.var(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(0, 5) }), where: np.array(new[] { true, false, true })),
                "float64", new long[0], 2.5);
            var o = np.zeros(new Shape(3));
            ReferenceEquals(np.var(F(), 0, @out: o, where: M2()), o).Should().BeTrue();
            Expect(o, "float64", new long[] { 3 }, 0, 0, 21.390625);
            new Action(() => np.std(F(), 0, @out: np.zeros(new Shape(3), np.int32), where: M2()))
                .Should().Throw<ArgumentException>().WithMessage("Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     <c>mean=</c>: subtracted as given (NumPy neither divides nor casts it), so the deviations take <c>a - mean</c>'s
        ///     dtype — a weak C# literal adopts the array's dtype (int32 deviations wrap when squared in int8), a strong array
        ///     promotes —, and a mean that broadcasts to a larger shape makes the result larger.
        /// </summary>
        [TestMethod]
        public void VarStd_Mean_UsedAsGiven()
        {
            Expect(np.var(F(), 1, mean: np.mean(F(), 1, keepdims: true)), "float64", new long[] { 2 }, 4.763888888888889, 17.166666666666668);
            Expect(np.var(F(), 1), "float64", new long[] { 2 }, 4.763888888888889, 17.166666666666668);
            Expect(np.var(F(), mean: 0), "float64", new long[0], 11.510416666666666);
            Expect(np.var(A(), mean: 2), "float64", new long[0], 114);
            Expect(np.var(A(), mean: 2.5), "float64", new long[0], 104.25);
            Expect(np.var(A(), mean: NDArray.Scalar(2.5f)), "float64", new long[0], 104.25);
            Expect(np.var(F().astype(np.float32), 1, mean: np.array(new[,] { { 1.0 }, { 2.0 } })), "float64", new long[] { 2 }, 4.770833333333333, 23.416666666666668);
            Expect(np.var(F().astype(np.float32), 1, mean: 1.0), "float32", new long[] { 2 }, 4.770833492279053, 19.41666603088379);
            var big = np.var(F(), 1, mean: np.zeros(new Shape(3, 2, 3)));
            Expect(big, "float64", new long[] { 3, 3 },
                6.083333333333333, 1.4166666666666667, 15.520833333333334,
                6.083333333333333, 1.4166666666666667, 15.520833333333334,
                6.083333333333333, 1.4166666666666667, 15.520833333333334);
            Expect(np.var(np.array(new sbyte[] { 100, -100 }), mean: 0), "float64", new long[0], 16);
            Expect(np.var(F(), 1, where: M2(), mean: np.array(new[,] { { 1.0 }, { 2.0 } })), "float64", new long[] { 2 }, 2.65625, 33.125);
            Expect(np.var(np.array(new[] { true, false }), mean: 0), "float64", new long[0], 0.5);
            Expect(np.var(np.array(new[] { new Complex(1, 2), new Complex(3, -1) }), mean: new Complex(0, 1)), "float64", new long[0], 7.5);
            Expect(np.std(F(), 0, mean: np.zeros(new Shape(3))), "float64", new long[] { 3 }, 3.020761493398643, 1.4577379737113252, 4.8250647664046955);
            var o32 = np.zeros(new Shape(3), np.float32);
            np.var(F(), 0, @out: o32, mean: np.zeros(new Shape(3)));
            Expect(o32, "float32", new long[] { 3 }, 9.125, 2.125, 23.28125);
            Expect(np.var(F(), 0, keepdims: true, mean: np.zeros(new Shape(3))), "float64", new long[] { 1, 3 }, 9.125, 2.125, 23.28125);
            Expect(np.var(np.array(3.0), mean: 1.0), "float64", new long[0], 4);
            new Action(() => np.var(F(), 1, mean: np.zeros(new Shape(2))))
                .Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,3) (2,) ");
            new Action(() => np.var(np.array(new[] { true, false }), mean: true))
                .Should().Throw<TypeError>().WithMessage("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
        }

        /// <summary>
        ///     <c>correction=</c>, fromnumeric's rule run before anything else: it replaces ddof, which must then be 0 (a ddof of
        ///     0 together with a correction is allowed; a nonzero one is refused even before a bad axis).
        /// </summary>
        [TestMethod]
        public void VarStd_Correction()
        {
            Expect(np.var(F(), correction: 1), "float64", new long[0], 13.760416666666666);
            Expect(np.var(F(), ddof: 1), "float64", new long[0], 13.760416666666666);
            Expect(np.var(F(), ddof: 0, correction: 2), "float64", new long[0], 17.200520833333332);
            Expect(np.std(F(), 0, correction: 1), "float64", new long[] { 3 }, 1.7677669529663689, 1.7677669529663689, 6.540737725975565);
            new Action(() => np.var(F(), ddof: 1, correction: 1)).Should().Throw<ValueError>().WithMessage("ddof and correction can't be provided simultaneously.");
            new Action(() => np.var(F(), ddof: 1, correction: 0)).Should().Throw<ValueError>().WithMessage("ddof and correction can't be provided simultaneously.");
            new Action(() => np.std(F(), 9, ddof: 1, correction: 1)).Should().Throw<ValueError>().WithMessage("ddof and correction can't be provided simultaneously.");
        }

        // ------------------------------------------------------------------------------------------------ ndarray methods

        /// <summary>The ndarray methods take the same keywords (positionally too: keepdims after out).</summary>
        [TestMethod]
        public void NdarrayMethods_TakeTheKeywords()
        {
            var f = F();
            Expect(f.sum(initial: 1, where: M2()), "float64", new long[0], 0.25);
            Expect(f.sum(0, null, null, false, 1, M2()), "float64", new long[] { 3 }, 2.5, 1.5, -1.75);
            Expect(f.prod(1, initial: 2, where: M2()), "float64", new long[] { 2 }, 9.75, -6.0);
            Expect(f.mean(where: M2()), "float64", new long[0], -0.1875);
            Expect(f.var(where: M2()), "float64", new long[0], 12.23046875);
            Expect(f.var(mean: 0), "float64", new long[0], 11.510416666666666);
            Expect(f.mean(0, null, null, true), "float64", new long[] { 1, 3 }, 2.75, -0.75, -1.375);
            Expect(f.var(0, null, null, 1, true), "float64", new long[] { 1, 3 }, 3.125, 3.125, 42.78125);
        }

        // ------------------------------------------------------------------------------------------------ numpy.matrix

        /// <summary>
        ///     A numpy.matrix operand: NumPy calls the matrix METHOD (<c>matrix.sum(axis, dtype, out)</c>, ...) with every keyword
        ///     the caller passed, and the method takes none of keepdims / initial / where / mean, so the first one passed is
        ///     its TypeError; correction= is folded into ddof first; a matrix out gets the method's answer, a plain out its
        ///     values and then <c>_collapse</c>'s AttributeError.
        /// </summary>
        [TestMethod]
        public void MatrixOperand_MethodKeywords()
        {
            NDArray m = np.matrix(np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
            new Action(() => np.sum(m, keepdims: true, initial: 1, where: true)).Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'keepdims'");
            new Action(() => np.sum(m, initial: 1, where: true)).Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'initial'");
            new Action(() => np.sum(m, where: true)).Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'where'");
            new Action(() => np.prod(m, initial: 1)).Should().Throw<TypeError>().WithMessage("matrix.prod() got an unexpected keyword argument 'initial'");
            new Action(() => np.mean(m, where: true)).Should().Throw<TypeError>().WithMessage("matrix.mean() got an unexpected keyword argument 'where'");
            new Action(() => np.var(m, mean: 0)).Should().Throw<TypeError>().WithMessage("matrix.var() got an unexpected keyword argument 'mean'");
            new Action(() => np.std(m, where: true, mean: 0)).Should().Throw<TypeError>().WithMessage("matrix.std() got an unexpected keyword argument 'where'");
            new Action(() => np.var(m, ddof: 1, correction: 1)).Should().Throw<ValueError>().WithMessage("ddof and correction can't be provided simultaneously.");
            Expect(np.var(m, correction: 1), "float64", new long[0], 1.6666666666666667);
            var s = np.std(m, 0, correction: 1);
            (s is NDMatrix).Should().BeTrue();
            Expect(s, "float64", new long[] { 1, 2 }, 1.4142135623730951, 1.4142135623730951);

            Expect(np.sum(m, @out: np.matrix(np.array(new[,] { { 0.0 } }))), "float64", new long[0], 10);
            var mo = np.matrix(np.array(new[,] { { 0.0, 0.0 } }));
            ReferenceEquals(np.sum(m, 0, @out: mo), mo).Should().BeTrue();
            Expect(mo, "float64", new long[] { 1, 2 }, 4, 6);
            var plain = np.zeros(new Shape(1, 2));
            new Action(() => np.sum(m, 0, @out: plain)).Should().Throw<AttributeError>().WithMessage("'numpy.ndarray' object has no attribute '_collapse'");
            Expect(plain, "float64", new long[] { 1, 2 }, 4, 6);
            new Action(() => np.sum(m, 0, @out: np.zeros(new Shape(2))))
                .Should().Throw<ValueError>().WithMessage("output parameter for reduction operation add has the wrong number of dimensions: Found 1 but expected 2 (must match the operand's when keepdims=True)");
        }
    }
}
