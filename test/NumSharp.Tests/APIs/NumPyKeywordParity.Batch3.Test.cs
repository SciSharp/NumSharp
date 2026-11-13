using System;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 3 (2026-11-11, docs/plans/API_PARITY_DIFF.md): ndarray.astype in NumPy's keyword order; the
    ///     <c>_like</c> family's <c>order='K'</c> and <c>shape=</c>; np.nansum / nanprod / nanmean / nanvar / nanstd with
    ///     NumPy's whole signature; ONE signature for the ten unary ufuncs, np.ravel / ndarray.ravel and np.repeat /
    ///     ndarray.repeat (the owner kept the generic scalar form with <c>NDArray repeats</c>); np.hstack / np.vstack's
    ///     <c>dtype</c> / <c>casting</c> beside the <c>params</c> form the owner kept. Every expected value or text is NumPy
    ///     2.4.2's answer for the same call (probed 2026-11-11, batch3_numpy_evidence.py).
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch3Tests
    {
        /// <summary>The public static overloads of <c>np.<paramref name="name"/></c>.</summary>
        /// <param name="name">The function name.</param>
        /// <returns>Every public static method of that name on <see cref="np"/>.</returns>
        private static MethodInfo[] NpForms(string name)
            // Without the kept int / int[] / int? forwards of an NDAxis member (the owner's batch-6 decision).
            => typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).WithoutKeptIntegerForms();

        /// <summary>The public instance members of <paramref name="type"/> named <paramref name="name"/> that the type itself
        ///     declares.</summary>
        /// <param name="type">The array class.</param>
        /// <param name="name">The method name.</param>
        /// <returns>The declared overloads.</returns>
        private static MethodInfo[] Own(Type type, string name)
            => type.GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == name && m.DeclaringType == type).ToArray();

        /// <summary>The parameter names of <paramref name="method"/>, in order.</summary>
        /// <param name="method">The method.</param>
        /// <returns>Its parameter names.</returns>
        private static string[] Names(MethodInfo method) => method.GetParameters().Select(p => p.Name).ToArray();

        /// <summary><c>[[1, nan, 3], [nan, 5, 6]]</c>, the nan reductions' operand.</summary>
        private static NDArray N => np.array(new double[,] { { 1, double.NaN, 3 }, { double.NaN, 5, 6 } });

        // ------------------------------------------------------------------ fix 1: astype

        /// <summary>astype is one member in NumPy's order (dtype, order, casting, copy) on every array class that declares it.</summary>
        [TestMethod]
        public void Astype_OneSignature_InNumPysOrder()
        {
            foreach (var type in new[] { typeof(NDArray), typeof(NDMatrix), typeof(NDCharArray), typeof(NDMaskedArray) })
            {
                var forms = Own(type, "astype");
                forms.Should().HaveCount(1, type.Name);
                Names(forms[0]).Should().Equal(new[] { "dtype", "order", "casting", "copy" }, type.Name);
            }
        }

        /// <summary>NumPy: <c>a.astype(np.float32, 'F')</c> is F-contiguous, <c>a.astype(np.int32, 'K', 'safe')</c> is the
        ///     safe-cast refusal, and <c>copy=False</c> with a satisfied order returns the array itself.</summary>
        [TestMethod]
        public void Astype_PositionalOrderAndCasting_AsNumPy()
        {
            var a = np.arange(6.0).reshape(2, 3);
            a.astype(np.float32, 'F').Shape.IsFContiguous.Should().BeTrue();
            Action safe = () => a.astype(np.int32, 'K', "safe");
            safe.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'");
            ReferenceEquals(a.astype(np.float64, copy: false), a).Should().BeTrue();
            ReferenceEquals(a.astype(np.float64, 'C', "unsafe", false), a).Should().BeTrue();
            ReferenceEquals(a.astype(np.float64, 'F', "unsafe", false), a).Should().BeFalse();   // the order is not satisfied
        }

        /// <summary>MaskedArray.astype is ndarray's: the order lays the data out, the mask follows the new data's layout
        ///     (NumPy's __array_finalize__ casts it in that order), and the casting rule refuses before anything else.</summary>
        [TestMethod]
        public void MaskedAstype_OrderCastingCopy_AsNumPy()
        {
            var m = np.ma.array(np.arange(6.0).reshape(2, 3), mask: np.array(new[,] { { false, true, false }, { false, false, true } }));
            var f = m.astype(np.float32, 'F');
            f.data.Shape.IsFContiguous.Should().BeTrue();
            f.mask.Shape.IsFContiguous.Should().BeTrue();
            f.mask.ToArray<bool>().Should().Equal(false, true, false, false, false, true);
            Action safe = () => m.astype(np.int32, casting: "safe");
            safe.Should().Throw<InvalidCastException>().WithMessage("*according to the rule 'safe'");
            ReferenceEquals(m.astype(np.float64, copy: false), m).Should().BeTrue();
        }

        // ------------------------------------------------------------------ fix 2: the _like family

        /// <summary>Each _like function is one member in NumPy's order, order defaulting to 'K' and shape to None.</summary>
        [TestMethod]
        public void Like_OneSignature_InNumPysOrder()
        {
            NpForms("zeros_like").Should().ContainSingle().Which.Should().Match<MethodInfo>(m =>
                Names(m).SequenceEqual(new[] { "a", "dtype", "order", "shape", "device" }));
            NpForms("ones_like").Should().ContainSingle().Which.Should().Match<MethodInfo>(m =>
                Names(m).SequenceEqual(new[] { "a", "dtype", "order", "shape", "device" }));
            NpForms("empty_like").Should().ContainSingle().Which.Should().Match<MethodInfo>(m =>
                Names(m).SequenceEqual(new[] { "prototype", "dtype", "order", "shape", "device" }));
            NpForms("full_like").Should().ContainSingle().Which.Should().Match<MethodInfo>(m =>
                Names(m).SequenceEqual(new[] { "a", "fill_value", "dtype", "order", "shape", "device" }));
        }

        /// <summary>NumPy's PyArray_NewLikeArrayWithShape layouts: 'K' keeps an F prototype's layout for a shape of the same
        ///     rank, C for another rank, and a transposed prototype's axis permutation for the same rank.</summary>
        [TestMethod]
        public void Like_ShapeOverride_KLayouts_AsNumPy()
        {
            var f = np.asfortranarray(np.arange(6.0).reshape(2, 3));
            np.zeros_like(f, shape: (3, 2)).Shape.IsFContiguous.Should().BeTrue();
            np.zeros_like(f, shape: (3, 2)).Shape.IsContiguous.Should().BeFalse();
            var c3 = np.zeros_like(f, shape: (2, 3, 1));
            c3.Shape.IsContiguous.Should().BeTrue();
            c3.Shape.IsFContiguous.Should().BeFalse();
            var t = np.arange(24.0).reshape(2, 3, 4).transpose(2, 0, 1);
            np.zeros_like(t, shape: (4, 2, 3)).strides.Should().Equal(8, 96, 32);
            np.zeros_like(t, shape: (4, 3, 2)).strides.Should().Equal(8, 64, 32);
        }

        /// <summary>NumPy: <c>ones_like(a, order='F', shape=(2, 2))</c>, <c>full_like(a, 7.9, dtype=int, shape=(2,))</c>
        ///     is [7, 7], <c>shape=()</c> is 0-d, <c>shape=None</c> is the prototype's, a negative dimension raises.</summary>
        [TestMethod]
        public void Like_ShapeAndOrderKeywords_AsNumPy()
        {
            var a = np.arange(6.0).reshape(2, 3);
            var o = np.ones_like(a, order: 'F', shape: (2, 2));
            o.shape.Should().Equal(2L, 2L);
            o.Shape.IsFContiguous.Should().BeTrue();
            o.ToArray<double>().Should().AllBeEquivalentTo(1.0);
            np.full_like(a, 7, shape: new[] { 4 }).ToArray<double>().Should().Equal(7.0, 7.0, 7.0, 7.0);
            var i = np.full_like(a, 7.9, np.int64, shape: 2);
            i.dtype.Should().Be(np.int64);
            i.ToArray<long>().Should().Equal(7L, 7L);
            np.zeros_like(a, shape: Shape.NewScalar()).ndim.Should().Be(0);
            np.empty_like(a, shape: null).shape.Should().Equal(2L, 3L);
            Action negative = () => np.zeros_like(a, shape: new[] { -1 });
            negative.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        // ------------------------------------------------------------------ fix 3: nansum / nanprod

        /// <summary>np.nansum / np.nanprod take NumPy's whole signature in NumPy's order.</summary>
        [TestMethod]
        public void NanSumProd_NumPysSignature()
        {
            foreach (var name in new[] { "nansum", "nanprod" })
                Names(NpForms(name).Single()).Should().Equal(new[] { "a", "axis", "dtype", "out", "keepdims", "initial", "where" }, name);
        }

        /// <summary>NumPy: nansum(n, 0, float32) is float32 [1, 5, 9], initial=10 gives 25, where= masks, out= is
        ///     returned, a narrow dtype wraps (int8 [100, 100] is -56); nanprod's initial / where likewise.</summary>
        [TestMethod]
        public void NanSumProd_Keywords_AsNumPy()
        {
            var s = np.nansum(N, 0, np.float32);
            s.dtype.Should().Be(np.float32);
            s.ToArray<float>().Should().Equal(1f, 5f, 9f);
            ((double)np.nansum(N, initial: 10)).Should().Be(25.0);
            np.nansum(N, 1, where: np.array(new[,] { { true, false, true }, { true, true, false } })).ToArray<double>().Should().Equal(4.0, 5.0);
            var o = np.zeros(new Shape(3));
            ReferenceEquals(np.nansum(N, 0, @out: o), o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(1.0, 5.0, 9.0);
            ((sbyte)np.nansum(np.array(new sbyte[] { 100, 100 }), dtype: np.int8)).Should().Be(-56);
            ((double)np.nanprod(N, initial: 2)).Should().Be(180.0);
            np.nanprod(N, 0, where: np.array(new[] { true, false, true })).ToArray<double>().Should().Equal(1.0, 1.0, 18.0);
            np.nansum(N, 0, null, null, true).shape.Should().Equal(1L, 3L);   // keepdims is NumPy's fifth argument
        }

        // ------------------------------------------------------------------ fix 4: nanmean / nanvar / nanstd

        /// <summary>np.nanmean / nanvar / nanstd take NumPy's whole signature in NumPy's order.</summary>
        [TestMethod]
        public void NanMeanVarStd_NumPysSignature()
        {
            Names(NpForms("nanmean").Single()).Should().Equal("a", "axis", "dtype", "out", "keepdims", "where");
            foreach (var name in new[] { "nanvar", "nanstd" })
                Names(NpForms(name).Single()).Should().Equal(
                    new[] { "a", "axis", "dtype", "out", "ddof", "keepdims", "where", "mean", "correction" }, name);
        }

        /// <summary>NumPy: nanmean(n, 1, float32) is float32 [2, 5.5]; where= masks the count and the sum; out= is returned;
        ///     a non-inexact out is "If a is inexact, then out must be inexact".</summary>
        [TestMethod]
        public void NanMean_Keywords_AsNumPy()
        {
            var m = np.nanmean(N, 1, np.float32);
            m.dtype.Should().Be(np.float32);
            m.ToArray<float>().Should().Equal(2f, 5.5f);
            np.nanmean(N, 1, where: np.array(new[,] { { true, true, false }, { true, true, true } })).ToArray<double>().Should().Equal(1.0, 5.5);
            var o = np.zeros(new Shape(2));
            ReferenceEquals(np.nanmean(N, 1, @out: o), o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(2.0, 5.5);
            Action intOut = () => np.nanmean(N, 1, @out: np.zeros(new Shape(2), np.int64));
            intOut.Should().Throw<TypeError>().WithMessage("If a is inexact, then out must be inexact");
        }

        /// <summary>NumPy: nanvar(n, 1, correction=1) is [2, 0.5]; ddof with correction raises; mean= replaces the computed
        ///     mean ([1, 0.25]); nanstd(n, 0, float32, None, 1) is [nan, nan, 2.1213202]; where= masks; out= is returned; an
        ///     integer input's correction is np.var's (1.6666666666666667).</summary>
        [TestMethod]
        public void NanVarStd_Keywords_AsNumPy()
        {
            np.nanvar(N, 1, correction: 1).ToArray<double>().Should().Equal(2.0, 0.5);
            Action both = () => np.nanvar(N, 1, ddof: 1, correction: 1);
            both.Should().Throw<ValueError>().WithMessage("ddof and correction can't be provided simultaneously.");
            np.nanvar(N, 1, mean: np.nanmean(N, 1, keepdims: true)).ToArray<double>().Should().Equal(1.0, 0.25);
            var s = np.nanstd(N, 0, np.float32, null, 1).ToArray<float>();
            float.IsNaN(s[0]).Should().BeTrue();
            float.IsNaN(s[1]).Should().BeTrue();
            s[2].Should().Be(2.1213202f);
            ((double)np.nanstd(N, where: np.array(new[,] { { true, true, true }, { false, true, true } }))).Should().Be(1.920286436967152);
            var o = np.zeros(new Shape(3));
            ReferenceEquals(np.nanstd(N, 0, @out: o), o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(0.0, 0.0, 1.5);
            ((double)np.nanvar(np.array(new[,] { { 1, 2 }, { 3, 4 } }), correction: 1)).Should().BeApproximately(1.6666666666666667, 1e-15);
        }

        // ------------------------------------------------------------------ fix 5: the unary ufuncs

        /// <summary>The ten unary ufuncs keep NumPy's one ufunc-shaped signature; a one-argument call binds it.</summary>
        [TestMethod]
        public void UnaryUfuncs_OneSignature()
        {
            foreach (var name in new[] { "abs", "absolute", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "square" })
            {
                var forms = NpForms(name);
                forms.Should().HaveCount(1, name);
                Names(forms[0]).Skip(1).Should().Equal(new[] { "out", "where", "dtype" }, name);
            }
            np.abs(np.array(new sbyte[] { -128, 5 })).ToArray<sbyte>().Should().Equal((sbyte)-128, (sbyte)5);   // NumPy wraps
            np.square(np.array(new sbyte[] { 12 })).ToArray<sbyte>().Should().Equal((sbyte)-112);
        }

        // ------------------------------------------------------------------ fix 8: ravel

        /// <summary>np.ravel and ndarray / matrix / chararray .ravel are one member each with NumPy's order='C'.</summary>
        [TestMethod]
        public void Ravel_OneSignature_DefaultC()
        {
            NpForms("ravel").Should().ContainSingle();
            foreach (var type in new[] { typeof(NDArray), typeof(NDMatrix), typeof(NDCharArray) })
                Own(type, "ravel").Should().ContainSingle(type.Name);
            var f = np.asfortranarray(np.arange(6).reshape(2, 3));
            np.ravel(f).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            f.ravel().ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.ravel(f, 'F').ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
        }

        // ------------------------------------------------------------------ fix 10: repeat

        /// <summary>np.repeat is NumPy's one signature plus the owner's generic scalar form with an NDArray count;
        ///     ndarray.repeat / matrix.repeat take the count as an array.</summary>
        [TestMethod]
        public void Repeat_Signatures()
        {
            var forms = NpForms("repeat");
            forms.Should().HaveCount(2);
            forms.Single(m => !m.IsGenericMethod).GetParameters().Select(p => p.ParameterType).Should()
                .Equal(typeof(NDArray), typeof(NDArray), typeof(int?));
            forms.Single(m => m.IsGenericMethod).GetParameters()[1].ParameterType.Should().Be(typeof(NDArray));
            Own(typeof(NDArray), "repeat").Should().ContainSingle();
            Own(typeof(NDMatrix), "repeat").Should().ContainSingle();
        }

        /// <summary>NumPy: np.repeat(5, 3) is [5, 5, 5], a size-1 count broadcasts, two counts for one element do not, a
        ///     matrix operand stays a matrix, and np.repeat(a, 2) / a.repeat(2) tile each element.</summary>
        [TestMethod]
        public void Repeat_Values_AsNumPy()
        {
            np.repeat(5, 3).ToArray<int>().Should().Equal(5, 5, 5);
            np.repeat(5, new[] { 2 }).ToArray<int>().Should().Equal(5, 5);
            Action two = () => np.repeat(5, new[] { 1, 2 });
            two.Should().Throw<Exception>().WithMessage("operands could not be broadcast together with shape (1,) (2,)");
            var m = np.repeat(np.asmatrix(np.array(new[,] { { 1, 2 } })), 2);
            m.Should().BeOfType<NDMatrix>();
            m.shape.Should().Equal(1L, 4L);
            m.ToArray<int>().Should().Equal(1, 1, 2, 2);
            np.repeat(np.array(new[] { 1, 2, 3 }), 2).ToArray<int>().Should().Equal(1, 1, 2, 2, 3, 3);
            np.array(new[] { 1, 2 }).repeat(new[] { 2, 1 }).ToArray<int>().Should().Equal(1, 1, 2);
        }

        // ------------------------------------------------------------------ fix 11: hstack / vstack

        /// <summary>np.hstack / np.vstack keep the params form beside NumPy's (tup, dtype, casting); np.vstack and its
        ///     deprecated alias np.row_stack also take np.char.split's object arrays of lists (one <c>NDListArray[]</c>
        ///     sequence form, API-parity batch 12).</summary>
        [TestMethod]
        public void HVStack_Signatures()
        {
            foreach (var name in new[] { "hstack", "vstack", "row_stack" })
            {
                var forms = NpForms(name);
                // The NDListArray[] form is vstack's (and row_stack's) third, typed by its sequence of list arrays.
                var listForms = forms.Where(m => m.GetParameters().Length == 1
                                                 && m.GetParameters()[0].ParameterType == typeof(NDListArray[])).ToList();
                listForms.Should().HaveCount(name == "hstack" ? 0 : 1, name);
                forms.Should().HaveCount(2 + listForms.Count, name);
                forms.Should().Contain(m => Names(m).SequenceEqual(new[] { "tup", "dtype", "casting" }), name);
                forms.Should().Contain(m => m.GetParameters().Length == 1 && m.GetParameters()[0].GetCustomAttribute<ParamArrayAttribute>() != null, name);
            }
        }

        /// <summary>NumPy: hstack([b, b], dtype=float32) is float32 (6,); vstack(..., dtype=int8, casting='unsafe') is int8
        ///     (2, 3); casting='safe' refuses int64 -> int8 with NumPy's text; np.hstack(b, b) still stacks.</summary>
        [TestMethod]
        public void HVStack_DtypeCasting_AsNumPy()
        {
            var b = np.arange(3);
            var h = np.hstack(new[] { b, b }, dtype: np.float32);
            h.dtype.Should().Be(np.float32);
            h.ToArray<float>().Should().Equal(0f, 1f, 2f, 0f, 1f, 2f);
            var v = np.vstack(new[] { b, b }, np.int8, "unsafe");
            v.dtype.Should().Be(np.int8);
            v.shape.Should().Equal(2L, 3L);
            Action safe = () => np.vstack(new[] { b, b }, np.int8, "safe");
            safe.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('int64') to dtype('int8') according to the rule 'safe'");
            np.hstack(b, b).shape.Should().Equal(6L);
            np.hstack(new[] { b, b }).shape.Should().Equal(6L);
        }
    }
}
