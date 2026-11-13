using System;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 6 (2026-11-12, docs/plans/API_PARITY_DIFF.md): NumPy's int-or-tuple axis for np.fft.fftshift /
    ///     ifftshift, np.linalg.norm / vector_norm and np.apply_over_axes; NumPy's axis type (a tuple refused in NumPy's
    ///     order) for np.nanargmax / nanargmin and chararray.argmax / argmin; one Python-int signature for np.base_repr /
    ///     binary_repr; one count type for logspace / geomspace and one window-length type for fftfreq / rfftfreq; np.unwrap
    ///     in NumPy's order with an NDArray period; dtype.newbyteorder's str; the legacy samplers' <c>size=None</c> and
    ///     random_sample's one size - and the owner's decision to keep the <c>int</c> / <c>int[]</c> overloads beside every
    ///     NumPy-axis member (<c>int?</c> for a one-integer axis). Every expected value, dtype or text is NumPy 2.4.2's answer
    ///     for the same call (probed 2026-11-12: batch6_numpy_evidence.py and the period / interp / permutation probes).
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch6Tests
    {
        /// <summary><c>np.arange(6).reshape(2, 3)</c> (int64), the evidence's <c>a2</c>.</summary>
        private static NDArray A2 => np.arange(6L).reshape(2, 3);

        /// <summary><c>np.arange(24.0).reshape(2, 3, 4)</c>, the evidence's <c>a3</c>.</summary>
        private static NDArray A3 => np.arange(24.0).reshape(2, 3, 4);

        /// <summary>The 2 x 3 float array with one NaN per row, the evidence's <c>f</c>.</summary>
        private static NDArray F => np.array(new double[,] { { 1, double.NaN, 3 }, { 4, 5, double.NaN } });

        /// <summary>The 2 x 2 float array whose first row is all NaN, the evidence's <c>fn</c>.</summary>
        private static NDArray FN => np.array(new double[,] { { double.NaN, double.NaN }, { 1, 2 } });

        /// <summary>The int64 signal <c>[0, 1, 2, -1, 0]</c>, the evidence's <c>p</c>.</summary>
        private static NDArray P => np.array(new long[] { 0, 1, 2, -1, 0 });

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 1: np.fft.fftshift / ifftshift
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>fftshift / ifftshift over NumPy's axes forms: the values, the IndexError past the rank, a subclass
        ///     operand's plain result and roll's F layout.</summary>
        [TestMethod]
        public void FftShift_AxesForms_AsNumPy()
        {
            np.fft.fftshift(np.arange(5L)).ToArray<long>().Should().Equal(3L, 4L, 0L, 1L, 2L);
            np.fft.fftshift(A2).ToArray<long>().Should().Equal(5L, 3L, 4L, 2L, 0L, 1L);
            np.fft.fftshift(A2, 1).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.fft.fftshift(A2, -1).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.fft.fftshift(A2, (0, 1)).ToArray<long>().Should().Equal(5L, 3L, 4L, 2L, 0L, 1L);
            np.fft.fftshift(A2, new[] { 1 }).ToArray<long>().Should().Equal(2L, 0L, 1L, 5L, 3L, 4L);
            np.fft.fftshift(A2, (1, 1)).ToArray<long>().Should().Equal(1L, 2L, 0L, 4L, 5L, 3L);
            np.fft.fftshift(A2, new NDAxis()).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            foreach (var bad in new NDAxis[] { 2, (0, 5), -3 })
            {
                Action act = () => np.fft.fftshift(A2, bad);
                act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
            }

            np.fft.ifftshift(np.arange(5L)).ToArray<long>().Should().Equal(2L, 3L, 4L, 0L, 1L);
            np.fft.ifftshift(A2, 1).ToArray<long>().Should().Equal(1L, 2L, 0L, 4L, 5L, 3L);
            np.fft.ifftshift(A2, (0, 1)).ToArray<long>().Should().Equal(4L, 5L, 3L, 1L, 2L, 0L);

            var m = np.asmatrix(np.array(new long[,] { { 1, 2 }, { 3, 4 } }));
            var shifted = np.fft.fftshift(m);
            shifted.Should().NotBeOfType<NDMatrix>("NumPy's asarray drops the matrix class");
            shifted.ToArray<long>().Should().Equal(4L, 3L, 2L, 1L);
            np.fft.fftshift(np.asfortranarray(A2)).Shape.IsFContiguous.Should().BeTrue("roll's empty_like keeps F order");
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fixes 3 and 4: np.linalg.norm / vector_norm
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>norm's axis: an int is NumPy's 1-tuple, a 2-tuple the matrix branch, other lengths and repeats NumPy's
        ///     ValueErrors, and the all-None call (ambiguous before) binds.</summary>
        [TestMethod]
        public void Norm_AxisForms_AsNumPy()
        {
            var row = new[] { 8.94427190999916, 10.344080432788601, 11.832159566199232, 13.379088160259652 };
            np.linalg.norm(A3, null, 1)["0"].ToArray<double>().Should().Equal(row);
            np.linalg.norm(A3, null, new[] { 1 })["0"].ToArray<double>().Should().Equal(row);
            np.linalg.norm(A3, null, (0, 2)).ToArray<double>().Should().Equal(27.349588662354687, 36.823905279043935, 47.11687595755899);
            np.linalg.norm(A3, 1, (0, 2)).ToArray<double>().Should().Equal(18.0, 26.0, 34.0);
            np.linalg.norm(A3, null, null, false).GetDouble().Should().Be(65.75712889109438);
            np.linalg.norm(A3, axis: new[] { 1 }, keepdims: true).shape.Should().Equal(2L, 1L, 4L);
            foreach (var bad in new NDAxis[] { new NDAxis(), (0, 1, 2) })
            {
                Action act = () => np.linalg.norm(A3, null, bad);
                act.Should().Throw<ValueError>().WithMessage("Improper number of dimensions to norm.");
            }
            Action twice = () => np.linalg.norm(A3, null, (0, 0));
            twice.Should().Throw<ValueError>().WithMessage("Duplicate axes given.");
            Action past = () => np.linalg.norm(A3, null, (0, 3));
            past.Should().Throw<AxisError>().WithMessage("axis 3 is out of bounds for array of dimension 3*");
        }

        /// <summary>vector_norm's axis: an int is NumPy's integer branch, any tuple a vector norm over the set, <c>()</c> the
        ///     elementwise magnitude, a repeat NumPy's "repeated axis".</summary>
        [TestMethod]
        public void VectorNorm_AxisForms_AsNumPy()
        {
            np.linalg.vector_norm(A3).GetDouble().Should().Be(65.75712889109438);
            np.linalg.vector_norm(A3, 1)["1"].ToArray<double>().Should().Equal(28.284271247461902, 29.9833287011299, 31.68595903550972, 33.391615714128);
            np.linalg.vector_norm(A3, new[] { 1 })["1"].ToArray<double>().Should().Equal(28.284271247461902, 29.9833287011299, 31.68595903550972, 33.391615714128);
            np.linalg.vector_norm(A3, (0, 2)).ToArray<double>().Should().Equal(27.349588662354687, 36.823905279043935, 47.11687595755899);
            np.linalg.vector_norm(A3, (0, 1, 2)).GetDouble().Should().Be(65.75712889109438);
            np.linalg.vector_norm(A3, new NDAxis()).shape.Should().Equal(2L, 3L, 4L);
            np.linalg.vector_norm(A3, (0, 2), keepdims: true).shape.Should().Equal(1L, 3L, 1L);
            Action twice = () => np.linalg.vector_norm(A3, (0, 0));
            twice.Should().Throw<ValueError>().WithMessage("repeated axis");
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 5: np.apply_over_axes
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>apply_over_axes' axes: an int is the 1-tuple, a tuple applies func per item in order (a repeat applies it
        ///     again), and <c>()</c> returns the input itself.</summary>
        [TestMethod]
        public void ApplyOverAxes_Axes_AsNumPy()
        {
            Func<NDArray, int, NDArray> sum = (x, ax) => np.sum(x, ax);
            var one = np.apply_over_axes(sum, A3, 1);
            one.shape.Should().Equal(2L, 1L, 4L);
            one.ToArray<double>().Should().Equal(12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0);
            var set = np.apply_over_axes(sum, A3, (0, 2));
            set.shape.Should().Equal(1L, 3L, 1L);
            set.ToArray<double>().Should().Equal(60.0, 92.0, 124.0);
            np.apply_over_axes(sum, A3, new[] { 0, 2 }).ToArray<double>().Should().Equal(60.0, 92.0, 124.0);
            np.apply_over_axes(sum, A3, -1).shape.Should().Equal(2L, 3L, 1L);
            var a = A3;
            ReferenceEquals(np.apply_over_axes(sum, a, new NDAxis()), a).Should().BeTrue();
            np.apply_over_axes(sum, A3, (0, 0)).shape.Should().Equal(1L, 3L, 4L);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fixes 6 and 7: nanargmax / nanargmin and chararray.argmax / argmin with a tuple
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>nanargmax / nanargmin refuse a tuple in NumPy's order: the NaN mask's np.all over the tuple first
        ///     (AxisError, a repeat, an all-NaN slice), then argmax's TypeError - and no mask at all for an integer or an
        ///     empty array.</summary>
        [TestMethod]
        public void NanArgMax_Tuple_RefusedInNumPysOrder()
        {
            const string tuple = "'tuple' object cannot be interpreted as an integer";
            Action one = () => np.nanargmax(F, new[] { 1 });
            one.Should().Throw<TypeError>().WithMessage(tuple);
            Action two = () => np.nanargmax(F, (0, 1));
            two.Should().Throw<TypeError>().WithMessage(tuple);
            Action none = () => np.nanargmax(F, new NDAxis());
            none.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered");
            Action allNan = () => np.nanargmax(FN, new[] { 1 });
            allNan.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered");
            Action past = () => np.nanargmax(F, (0, 5));
            past.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            Action repeat = () => np.nanargmax(F, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action integer = () => np.nanargmax(np.array(new long[,] { { 1, 5, 3 }, { 4, 2, 6 } }), (0, 5));
            integer.Should().Throw<TypeError>().WithMessage(tuple);
            Action empty = () => np.nanargmax(np.zeros(new Shape(0, 2)), new[] { 0 });
            empty.Should().Throw<TypeError>().WithMessage(tuple);
            Action min = () => np.nanargmin(FN, new[] { 1 });
            min.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered");

            np.nanargmax(F, 1).ToArray<long>().Should().Equal(2L, 1L);
            np.nanargmax(F, (int?)1).ToArray<long>().Should().Equal(2L, 1L);
            np.nanargmax(F, null).GetInt64().Should().Be(4L);
            np.nanargmax(F).Should().Be(4L);
        }

        /// <summary>chararray.argmax / argmin: a tuple is ndarray.argmax's TypeError, first; every other call is NumPy's
        ///     finalize ValueError (the int64 result made as a chararray).</summary>
        [TestMethod]
        public void CharArrayArgMax_Tuple_IsNumPysTypeErrorFirst()
        {
            var c = np.@char.array(np.array(new[,] { { "b", "a" }, { "d", "c" } }));
            const string finalize = "Can only create a chararray from string data.";
            Action axis = () => c.argmax(1);
            axis.Should().Throw<ValueError>().WithMessage(finalize);
            Action tuple = () => c.argmax(new NDAxis(new[] { 1 }));
            tuple.Should().Throw<TypeError>().WithMessage("'tuple' object cannot be interpreted as an integer");
            Action flat = () => c.argmax();
            flat.Should().Throw<ValueError>().WithMessage(finalize);
            Action keep = () => c.argmin(1, keepdims: true);
            keep.Should().Throw<ValueError>().WithMessage(finalize);
            typeof(NDCharArray).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Where(m => m.Name == "argmax").Select(m => string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name)))
                .OrderBy(s => s).Should().Equal("", "Nullable`1, NDArray, Boolean", "Nullable`1, NDArray, Boolean");
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 8: base_repr / binary_repr
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>One Python-int signature: values past int64 are exact, every C# integer converts.</summary>
        [TestMethod]
        public void BaseRepr_BinaryRepr_PythonInt_AsNumPy()
        {
            np.base_repr(BigInteger.Pow(2, 70), 16).Should().Be("400000000000000000");
            np.base_repr(-BigInteger.Pow(2, 70), 36, 3).Should().Be("-0006X5KXTVUWILUKG");
            np.binary_repr(BigInteger.Pow(2, 70)).Should().Be("1" + new string('0', 70));
            np.binary_repr(-BigInteger.Pow(2, 65), 70).Should().Be("11111" + new string('0', 65));
            np.binary_repr(ulong.MaxValue).Should().Be(new string('1', 64));
            np.base_repr(5).Should().Be("101");
            foreach (var name in new[] { "base_repr", "binary_repr" })
                typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name)
                    .Select(m => m.GetParameters()[0].ParameterType).Should().Equal(new[] { typeof(BigInteger) }, name);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 10: logspace / geomspace
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>One count type, <c>long num = 50</c>; geomspace keeps its real / complex pair.</summary>
        [TestMethod]
        public void LogspaceGeomspace_OneCountType()
        {
            np.logspace(0, 2, 3).ToArray<double>().Should().Equal(1.0, 10.0, 100.0);
            np.logspace(0, 2).size.Should().Be(50);
            np.geomspace(1, 100, 3).ToArray<double>().Should().Equal(1.0, 10.0, 100.0);
            np.geomspace(new Complex(0, 1), new Complex(0, 100), 3).ToArray<Complex>()
                .Should().Equal(new Complex(0, 1), new Complex(0, 10), new Complex(0, 100));
            foreach (var name in new[] { "logspace", "geomspace" })
                typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name)
                    .Select(m => m.GetParameters().Single(p => p.Name == "num"))
                    .Should().OnlyContain(p => p.ParameterType == typeof(long) && (long)p.DefaultValue == 50L, name);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 11: np.unwrap
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's order: a positional second argument is discont (the old long overload made it the period).</summary>
        [TestMethod]
        public void Unwrap_PositionalSecondArgument_IsDiscont()
        {
            var r = np.unwrap(P, 4);
            r.dtype.Should().Be(np.float64);
            r.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, -1.0, 0.0);
            np.unwrap(np.array(new[] { 0.0, 1.0, 7.0 }), 4).ToArray<double>().Should().Equal(0.0, 1.0, 0.7168146928204138);
            typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == "unwrap")
                .Select(m => string.Join(", ", m.GetParameters().Select(p => p.Name)))
                .Should().Equal("p, discont, axis, period");
        }

        /// <summary>The period's kind picks NumPy's path: a C# integer is the weak Python int, a C# double the weak float,
        ///     a 0-d array or NumPy scalar a strong period whose dtype joins <c>result_type(diff(p), period)</c>.</summary>
        [TestMethod]
        public void Unwrap_PeriodKinds_AsNumPy()
        {
            var p8 = P.astype(np.int8);
            var pf32 = np.array(new float[] { 0, 1, 2, -1, 0 });
            var pf16 = pf32.astype(np.float16);
            void Is(NDArray r, NPTypeCode dtype, params double[] values)
            {
                r.typecode.Should().Be(dtype);
                r.astype(np.float64).ToArray<double>().Should().Equal(values);
            }

            Is(np.unwrap(P, period: 4), NPTypeCode.Int64, 0, 1, 2, 3, 4);
            Is(np.unwrap(P, period: 4.0), NPTypeCode.Double, 0, 1, 2, 3, 4);
            Is(np.unwrap(P, period: 5), NPTypeCode.Int64, 0, 1, 2, 4, 5);
            Is(np.unwrap(P, period: np.array(4L)), NPTypeCode.Int64, 0, 1, 2, 3, 4);
            Is(np.unwrap(P, period: np.array((sbyte)4)), NPTypeCode.Int64, 0, 1, 2, 3, 4);
            Is(np.unwrap(p8, period: np.array(4L)), NPTypeCode.Int64, 0, 1, 2, 3, 4);
            Is(np.unwrap(p8, period: 4), NPTypeCode.SByte, 0, 1, 2, 3, 4);
            Is(np.unwrap(p8, period: np.array(4f)), NPTypeCode.Single, 0, 1, 2, 3, 4);
            Is(np.unwrap(p8, period: 4.0), NPTypeCode.Double, 0, 1, 2, 3, 4);
            Is(np.unwrap(pf32, period: np.array(4.0)), NPTypeCode.Double, 0, 1, 2, 3, 4);
            Is(np.unwrap(pf32, period: 4.0), NPTypeCode.Single, 0, 1, 2, 3, 4);
            Is(np.unwrap(pf16, period: np.array(4f)), NPTypeCode.Single, 0, 1, 2, 3, 4);
            Is(np.unwrap(pf16, period: np.array(4L)), NPTypeCode.Double, 0, 1, 2, 3, 4);
            Is(np.unwrap(P, period: true), NPTypeCode.Int64, 0, 0, 0, 0, 0);
            Is(np.unwrap(P.astype(np.float64), period: true), NPTypeCode.Double, 0, 0, 0, 0, 0);
            Is(np.unwrap(np.array(new[] { true, false, true, true }), period: 4), NPTypeCode.Int64, 1, 0, 1, 1);
            Is(np.unwrap(P, period: np.array(new long[] { 4 })), NPTypeCode.Int64, 0, 1, 2, 3, 4);
            Is(np.unwrap(P, period: np.array(4UL)), NPTypeCode.Double, 0, 1, 2, 3, 4);
            Is(np.unwrap(P.astype(np.uint8), period: np.array(4L)), NPTypeCode.Int64, 0, 1, 2, 3, -252);
        }

        /// <summary>The period refusals in NumPy's words: an unsigned integer path's interval bound, a complex period, a text
        ///     period.</summary>
        [TestMethod]
        public void Unwrap_PeriodRefusals_AsNumPy()
        {
            Action unsigned = () => np.unwrap(P.astype(np.uint8), period: 4);
            unsigned.Should().Throw<OverflowError>().WithMessage("Python integer -2 out of bounds for uint8");
            Action complex = () => np.unwrap(P, period: new Complex(4, 0));
            complex.Should().Throw<TypeError>().WithMessage("ufunc 'remainder' not supported for the input types*");
            Action text = () => np.unwrap(P, period: "4");
            text.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for /: 'str' and 'int'");
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fix 13: dtype.newbyteorder
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy's new_order is a str: one signature, the char overload gone.</summary>
        [TestMethod]
        public void NewByteOrder_StrOnly_AsNumPy()
        {
            var d = np.dtype("<i4");
            d.newbyteorder().Should().Be(np.dtype(">i4"));
            d.newbyteorder(">").Should().Be(np.dtype(">i4"));
            d.newbyteorder("big").Should().Be(np.dtype(">i4"));
            Action empty = () => d.newbyteorder("");
            empty.Should().Throw<ValueError>().WithMessage("byteorder not recognized (got '')");
            typeof(DType).GetMethods().Where(m => m.Name == "newbyteorder").Select(m => m.GetParameters()[0].ParameterType)
                .Should().Equal(typeof(string));
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Fixes 14 and 15: the legacy np.random size
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary><c>size: null</c> is NumPy's size=None on the legacy samplers: one draw, NumPy's value (seed 0).</summary>
        [TestMethod]
        public void LegacySamplers_SizeNone_AsNumPy()
        {
            // A RandomState of its own per draw (the suite runs in parallel; the global np.random is shared): the same
            // stream as NumPy's np.random.seed(0).
            np.random.RandomState(0).normal(0, 1, size: null).GetDouble().Should().Be(1.764052345967664);
            np.random.RandomState(0).pareto(3.0, null).GetDouble().Should().Be(0.30381098483863833);
            np.random.RandomState(0).logseries(0.5, null).GetInt64().Should().Be(1L);
            np.random.RandomState(0).hypergeometric(5, 5, 3, null).GetInt64().Should().Be(2L);
            var viaArray = np.random.RandomState(0).pareto(3.0, new[] { 2, 3 }).ToArray<double>();
            np.random.RandomState(0).pareto(3.0, (2, 3)).ToArray<double>().Should().Equal(viaArray, "an int[] size binds Shape? with the same draws");
            new PCG64(0).random_raw(size: null).ndim.Should().Be(0);
        }

        /// <summary>The legacy surface: no non-nullable <c>Shape size</c> parameter is left, no priority-ranked size shim,
        ///     random_sample / random / ranf / sample take one <c>Shape?</c>, rand only loose dimensions.</summary>
        [TestMethod]
        public void LegacySamplers_SizeSurface()
        {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            var methods = typeof(NumPyRandom).GetMethods(flags);
            methods.Where(m => m.GetParameters().Any(p => p.Name == "size" && p.ParameterType == typeof(Shape)))
                .Select(m => m.Name).Should().BeEmpty("size=None needs a type that takes null");
            methods.Where(m => m.GetParameters().Any(p => p.Name == "size") &&
                               m.GetCustomAttributes().Any(a => a.GetType().Name == "OverloadResolutionPriorityAttribute"))
                .Select(m => m.Name).Should().BeEmpty("the int[] / long[] / long size shims are gone");
            foreach (var name in new[] { "random_sample", "random", "ranf", "sample" })
                methods.Where(m => m.Name == name).Select(m => m.GetParameters().Single().ParameterType)
                    .Should().Equal(new[] { typeof(Shape?) }, name);
            methods.Where(m => m.Name == "rand").Select(m => m.GetParameters().Single().ParameterType)
                .Should().Equal(new[] { typeof(long[]) });
            typeof(BitGenerator).GetMethod("random_raw").GetParameters()[0].ParameterType.Should().Be(typeof(Shape?));

            np.random.RandomState(0).random_sample().GetDouble().Should().Be(0.5488135039273248);
            np.random.RandomState(0).random_sample(null).GetDouble().Should().Be(0.5488135039273248);
            np.random.RandomState(0).random_sample((2, 3)).shape.Should().Equal(2L, 3L);
            np.random.RandomState(0).random_sample(5).shape.Should().Equal(5L);
            np.random.RandomState(0).rand(2, 3).shape.Should().Equal(2L, 3L);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // The owner's decision: the int / int[] overloads stay beside every NumPy-axis member
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Every public np / ndarray / chararray member that takes NumPy's axis as NDAxis has its kept integer forwards:
        ///     <c>int</c> and <c>int[]</c>, or <c>int?</c> for a one-integer axis (argmax / argmin / nanargmax / nanargmin).
        ///     The engine and numpy.matrix are outside the decision.
        /// </summary>
        [TestMethod]
        public void EveryNDAxisMember_KeepsItsIntegerOverloads()
        {
            var oneInteger = new[] { "argmax", "argmin", "nanargmax", "nanargmin" };
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            var owners = new[] { typeof(np), typeof(NDArray), typeof(NDCharArray), typeof(FourierModule), typeof(np.linalg) };
            var missing = new System.Collections.Generic.List<string>();
            int checkedMembers = 0;
            foreach (var owner in owners)
            {
                var all = owner.GetMethods(flags);
                foreach (var m in all.Where(m => m.GetParameters().Any(p => p.ParameterType == typeof(NDAxis) || p.ParameterType == typeof(NDAxis?))))
                {
                    checkedMembers++;
                    var wanted = oneInteger.Contains(m.Name) ? new[] { typeof(int?) } : new[] { typeof(int), typeof(int[]) };
                    foreach (var integer in wanted)
                    {
                        bool found = all.Any(s => s.Name == m.Name && s.GetParameters().Length == m.GetParameters().Length &&
                            s.GetParameters().Zip(m.GetParameters()).All(pair => pair.First.Name == pair.Second.Name &&
                                (pair.Second.ParameterType == typeof(NDAxis) || pair.Second.ParameterType == typeof(NDAxis?)
                                    ? pair.First.ParameterType == integer
                                    : pair.First.ParameterType == pair.Second.ParameterType)));
                        if (!found)
                            missing.Add($"{owner.Name}.{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))}) has no {integer.Name} form");
                    }
                }
            }
            checkedMembers.Should().BeGreaterThan(50);
            missing.Should().BeEmpty();
        }

        /// <summary>The kept forwards give the NDAxis form's answers - values, dtypes and the matrix arm.</summary>
        [TestMethod]
        public void KeptIntegerOverloads_AnswerAsTheNDAxisForm()
        {
            np.array_equal(np.sum(A3, 1), np.sum(A3, (NDAxis?)1)).Should().BeTrue();
            np.array_equal(np.sum(A3, new[] { 0, 2 }), np.sum(A3, (0, 2))).Should().BeTrue();
            np.array_equal(np.sum(A3, null), np.sum(A3)).Should().BeTrue();
            np.mean(A3, 1, keepdims: true).shape.Should().Equal(2L, 1L, 4L);
            np.argmax(A3, 1).ToArray<long>().Should().OnlyContain(i => i == 2L);
            np.argmax(A3, (int?)null).GetInt64().Should().Be(23L);
            np.moveaxis(A3, 0, -1).shape.Should().Equal(3L, 4L, 2L);
            np.moveaxis(A3, new[] { 0, 1 }, new[] { -1, -2 }).shape.Should().Equal(4L, 3L, 2L);
            np.median(A3, new[] { 0, 2 }).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            np.count_nonzero(A2, 1).ToArray<long>().Should().Equal(2L, 3L);
            var m = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            var s = np.sum(m, 0);
            s.Should().BeOfType<NDMatrix>("the int forward reaches the woven matrix arm");
            s.ToArray<double>().Should().Equal(4.0, 6.0);
        }
    }
}
