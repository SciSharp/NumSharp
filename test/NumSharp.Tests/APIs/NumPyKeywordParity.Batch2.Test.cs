using System;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 2 (2026-11-11, docs/plans/API_PARITY_DIFF.md): NumPy's keywords for np.stack (<c>out</c>,
    ///     <c>dtype</c>, <c>casting</c>), np.ldexp (<c>dtype</c>), ndarray.dot (<c>out</c>), numpy.ma's size (<c>obj</c>,
    ///     <c>axis</c>) and MaskedArray.clip (<c>min</c>, <c>max</c>); NumPy's tuple <c>out=(o1, o2)</c> for np.frexp /
    ///     np.modf; one signature for np.take, np.divmod, np.power, the shifts and np.promote_types; argsort without its
    ///     generic forms; MaskedArray.resize's ValueError. Every expected value or text is NumPy 2.4.2's answer for the same
    ///     call (probed 2026-11-11).
    /// </summary>
    /// <remarks>
    ///     <para>Two subclass rules bind np.frexp's / np.modf's <c>out</c> BY NAME (NumSharp.Build passes <c>default</c> for a
    ///     name the function lacks): <see cref="FrexpModf_MatrixOperand_PlainOutsComeBackThemselves"/> fails if the rules and
    ///     the public parameter part ways - the caller's out arrays would come back as matrix views.</para>
    ///     <para>Not changed, validated instead: ndarray.conj / conjugate keep their <c>out</c>. NumPy's C method parses one
    ///     optional POSITIONAL argument (<c>"|O&amp;:conjugate"</c>) while refusing it by keyword, so <c>a.conj(o) is o</c>;
    ///     <see cref="Conj_PositionalOut_IsNumPys"/> pins that.</para>
    /// </remarks>
    [TestClass]
    public class NumPyKeywordParityBatch2Tests
    {
        /// <summary>The public static overloads of <c>np.<paramref name="name"/></c>.</summary>
        /// <param name="name">The function name.</param>
        /// <returns>Every public static method of that name on <see cref="np"/>.</returns>
        private static MethodInfo[] NpForms(string name)
            => typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).ToArray();

        /// <summary>The parameter names of <paramref name="method"/>, in order.</summary>
        /// <param name="method">The method.</param>
        /// <returns>Its parameter names.</returns>
        private static string[] Names(MethodInfo method) => method.GetParameters().Select(p => p.Name).ToArray();

        /// <summary>The (2, 3) float64 operand <c>arange(6.0).reshape(2, 3)</c>.</summary>
        private static NDArray A => np.arange(6.0).reshape(2, 3);

        #region stack, ldexp, dot (fixes 1, 2, 4)

        /// <summary>
        ///     np.stack's NumPy keywords: <c>dtype=float32</c> casts, <c>out=</c> is written and returned (axis 1 too),
        ///     <c>casting='safe'</c> refuses float64 -> int32 with NumPy's text, <c>'unsafe'</c> casts, and <c>out</c> with
        ///     <c>dtype</c> is concatenate's TypeError.
        /// </summary>
        [TestMethod]
        public void Stack_OutDtypeCasting_AsNumPy()
        {
            var f = np.stack(new[] { A, A }, dtype: np.float32);
            f.dtype.Should().Be(np.float32);
            f.shape.Should().Equal(2L, 2L, 3L);
            var o = np.zeros(new Shape(2, 2, 3));
            ReferenceEquals(np.stack(new[] { A, A }, axis: 1, @out: o), o).Should().BeTrue();
            o.flatten().ToArray<double>().Should().Equal(0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5);
            // NumPy's text; NumPy's TYPE (TypeError) is OpenBugs.ApiParity_ConcatenateCastRefusal_IsNumPysTypeError's pin:
            // concatenate still raises InvalidCastException there.
            Action safe = () => np.stack(new[] { A, A }, dtype: np.int32, casting: "safe");
            safe.Should().Throw<Exception>()
                .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'");
            np.stack(new[] { A, A }, dtype: np.int32, casting: "unsafe").flatten().ToArray<int>()
              .Should().Equal(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5);
            Action both = () => np.stack(new[] { A, A }, @out: np.zeros(new Shape(2, 2, 3)), dtype: np.float64);
            both.Should().Throw<Exception>()   // NumPy's TypeError: the same OpenBugs pin (ArgumentException today)
                .WithMessage("concatenate() only takes `out` or `dtype` as an argument, but both were provided.");
            Names(NpForms("stack").Single()).Should().Equal("arrays", "axis", "out", "dtype", "casting");
        }

        /// <summary>
        ///     np.ldexp's <c>dtype=</c> fixes the loop's output and needs the operands to reach a float x int loop under
        ///     'safe' casting: float16 -> float32 and int8 -> float16 compute in the dtype; float64 -> float32, a uint64 or
        ///     float exponent and an int64 dtype have no loop (NumPy's signature text).
        /// </summary>
        [TestMethod]
        public void Ldexp_Dtype_PicksTheLoopAsNumPy()
        {
            var n = np.array(new long[] { 1, 3 });
            var f32 = np.ldexp(np.array(new[] { (Half)1.5, (Half)2.0 }), n, dtype: np.float32);
            f32.dtype.Should().Be(np.float32);
            f32.ToArray<float>().Should().Equal(3f, 16f);
            var f16 = np.ldexp(np.array(new sbyte[] { 3, 1 }), np.array(new long[] { 1, 2 }), dtype: np.float16);
            f16.dtype.Should().Be(np.float16);
            f16.ToArray<Half>().Should().Equal((Half)6, (Half)4);
            np.ldexp(np.array(new[] { 1.5, 2.0 }), n, dtype: np.float64).ToArray<double>().Should().Equal(3.0, 16.0);
            const string noLoop = "No loop matching the specified signature and casting was found for ufunc ldexp";
            foreach (var call in new Action[]
            {
                () => np.ldexp(np.array(new[] { 1.5, 2.0 }), n, dtype: np.float32),
                () => np.ldexp(np.array(new[] { 1.0 }), np.array(new ulong[] { 1 }), dtype: np.float64),
                () => np.ldexp(np.array(new[] { 1.0 }), np.array(new[] { 1.0 }), dtype: np.float64),
                () => np.ldexp(np.array(new[] { 1.0 }), np.array(new long[] { 1 }), dtype: np.int64),
            })
                call.Should().Throw<TypeError>().WithMessage(noLoop);
            Names(NpForms("ldexp").Single()).Should().Equal("x1", "x2", "out", "where", "dtype");
        }

        /// <summary>a.dot(b, out=o) writes the product into <c>o</c> and returns it; an out of another dtype is NumPy's
        ///     "output array is not acceptable".</summary>
        [TestMethod]
        public void Dot_Out_AsNumPy()
        {
            var o = np.zeros(new Shape(2, 2));
            ReferenceEquals(A.dot(np.arange(6.0).reshape(3, 2), @out: o), o).Should().BeTrue();
            o.flatten().ToArray<double>().Should().Equal(10.0, 13.0, 28.0, 40.0);
            Action wrong = () => A.dot(np.arange(6.0).reshape(3, 2), @out: np.zeros(new Shape(2, 2), np.float32));
            wrong.Should().Throw<ValueError>().WithMessage("output array is not acceptable*");
        }

        /// <summary>NumPy's <c>a.conj(o)</c> takes <c>o</c> by position and returns it; NumSharp's method keeps that
        ///     parameter (the X finding was refuted by the positional probe).</summary>
        [TestMethod]
        public void Conj_PositionalOut_IsNumPys()
        {
            var o = np.zeros(new Shape(1), np.complex128);
            ReferenceEquals(np.array(new[] { new Complex(1, 2) }).conj(o), o).Should().BeTrue();
            o.GetAtIndex<Complex>(0).Should().Be(new Complex(1, -2));
        }

        #endregion

        #region take, frexp / modf, one-signature ufuncs (fixes 5-9)

        /// <summary>np.take / a.take have one signature each; a C# integer index is a 0-d index (a 0-d result).</summary>
        [TestMethod]
        public void Take_OneSignature_LiteralIndexIsZeroD()
        {
            var r = np.take(np.arange(12.0).reshape(3, 4), 2);
            r.ndim.Should().Be(0);
            r.GetDouble().Should().Be(2.0);
            long variable = -1;
            np.take(np.arange(12.0).reshape(3, 4), variable, axis: 1).ToArray<double>().Should().Equal(3.0, 7.0, 11.0);
            Names(NpForms("take").Single()).Should().Equal("a", "indices", "axis", "out", "mode");
            typeof(NDArray).GetMethods().Where(m => m.Name == "take" && m.DeclaringType == typeof(NDArray)).Should().HaveCount(1);
        }

        /// <summary>
        ///     np.frexp / np.modf take NumPy's <c>out=(o1, o2)</c>: both outputs written and returned themselves, a null slot
        ///     allocated; frexp refuses every <c>dtype=</c> (no loop), modf computes in a float one.
        /// </summary>
        [TestMethod]
        public void FrexpModf_TupleOut_AsNumPy()
        {
            var x = np.array(new[] { 8.0, 0.5, 3.0 });
            var m = np.zeros(new Shape(3));
            var e = np.zeros(new Shape(3), np.int32);
            var (rm, re) = np.frexp(x, @out: (m, e));
            ReferenceEquals(rm, m).Should().BeTrue();
            ReferenceEquals(re, e).Should().BeTrue();
            m.ToArray<double>().Should().Equal(0.5, 0.5, 0.75);
            e.ToArray<int>().Should().Equal(4, 0, 2);
            var (freshM, givenE) = np.frexp(x, @out: (null, np.zeros(new Shape(3), np.int32)));
            freshM.ToArray<double>().Should().Equal(0.5, 0.5, 0.75);
            Action dtype = () => np.frexp(x, dtype: np.float64);
            dtype.Should().Throw<TypeError>()
                 .WithMessage("No loop matching the specified signature and casting was found for ufunc frexp");
            var f = np.zeros(new Shape(2));
            var i = np.zeros(new Shape(2));
            var (rf, ri) = np.modf(np.array(new[] { 1.5, -2.25 }), @out: (f, i));
            ReferenceEquals(rf, f).Should().BeTrue();
            f.ToArray<double>().Should().Equal(0.5, -0.25);
            i.ToArray<double>().Should().Equal(1.0, -2.0);
            np.modf(np.array(new[] { 1.5 }), dtype: np.float32).Fractional.dtype.Should().Be(np.float32);
            Names(NpForms("frexp").Single()).Should().Equal("x", "out", "where", "dtype");
            Names(NpForms("modf").Single()).Should().Equal("x", "out", "where", "dtype");
        }

        /// <summary>
        ///     A matrix input with plain outs: NumPy returns the out arrays themselves (and wraps only a fresh output in the
        ///     matrix class). Pins the by-name binding of NDSubclassRules.Frexp / Modf to the public <c>out</c>.
        /// </summary>
        [TestMethod]
        public void FrexpModf_MatrixOperand_PlainOutsComeBackThemselves()
        {
            var mx = np.asmatrix(np.array(new[,] { { 8.0, 0.5 } }));
            var mo = np.zeros(new Shape(1, 2));
            var eo = np.zeros(new Shape(1, 2), np.int32);
            var (rm, re) = np.frexp(mx, @out: (mo, eo));
            ReferenceEquals(rm, mo).Should().BeTrue();
            ReferenceEquals(re, eo).Should().BeTrue();
            var (freshM, freshE) = np.frexp(mx);
            freshM.Should().BeOfType<NDMatrix>();
            freshE.Should().BeOfType<NDMatrix>();
            var fo = np.zeros(new Shape(1, 2));
            var (rf, ri) = np.modf(mx, @out: (fo, null));
            ReferenceEquals(rf, fo).Should().BeTrue();
            ri.Should().BeOfType<NDMatrix>();
        }

        /// <summary>
        ///     divmod / power / left_shift / right_shift have one signature each, the ufunc form; a C# literal operand is
        ///     NumPy's weak Python int, so an int8 / uint8 operand keeps its dtype as in NumPy.
        /// </summary>
        [TestMethod]
        public void BinaryUfuncs_OneSignature_WeakLiteral()
        {
            var (q, r) = np.divmod(np.array(new sbyte[] { 7, 8 }), 3);
            q.dtype.Should().Be(np.int8);
            r.dtype.Should().Be(np.int8);
            var p = np.power(np.array(new sbyte[] { 2, 3 }), 2);
            p.dtype.Should().Be(np.int8);
            p.ToArray<sbyte>().Should().Equal((sbyte)4, (sbyte)9);
            np.left_shift(np.array(new sbyte[] { 1, 2 }), 2).ToArray<sbyte>().Should().Equal((sbyte)4, (sbyte)8);
            np.right_shift(np.array(new byte[] { 8, 4 }), 1).ToArray<byte>().Should().Equal((byte)4, (byte)2);
            foreach (var name in new[] { "divmod", "power", "left_shift", "right_shift" })
                NpForms(name).Should().HaveCount(1, name);
        }

        #endregion

        #region promote_types, argsort (fixes 10, 11)

        /// <summary>np.promote_types has NumPy's one signature; NPTypeCode, Type and dtype strings convert to DType and
        ///     answer as NumPy (int32 + float32 -> float64).</summary>
        [TestMethod]
        public void PromoteTypes_OneDTypeSignature()
        {
            np.promote_types(NPTypeCode.Int32, NPTypeCode.Single).Should().Be(np.float64);
            np.promote_types(typeof(int), typeof(float)).Should().Be(np.float64);
            np.promote_types("i4", "f4").Should().Be(np.float64);
            var form = NpForms("promote_types").Single();
            form.IsGenericMethodDefinition.Should().BeFalse();
            Names(form).Should().Equal("type1", "type2");
        }

        /// <summary>argsort has no generic form left (NumPy has none; the type argument was ignored anyway).</summary>
        [TestMethod]
        public void Argsort_NoGenericForms()
        {
            NpForms("argsort").Should().ContainSingle().Which.IsGenericMethodDefinition.Should().BeFalse();
            typeof(NDArray).GetMethods().Where(m => m.Name == "argsort" && m.DeclaringType == typeof(NDArray))
                .Should().ContainSingle().Which.IsGenericMethodDefinition.Should().BeFalse();
            np.argsort(np.array(new[] { 3, 1, 2 })).ToArray<long>().Should().Equal(1L, 2L, 0L);
        }

        #endregion

        #region numpy.ma (fixes 12-15)

        /// <summary>np.ma.default_fill_value: both forms take NumPy's <c>obj</c> and answer as NumPy (1e20, 999999, True,
        ///     (1e20+0j), '?' for None); the typed DType form stays by the owner's call.</summary>
        [TestMethod]
        public void DefaultFillValue_ObjKeyword_AsNumPy()
        {
            np.ma.default_fill_value(obj: np.float64).Should().Be(1e20);
            np.ma.default_fill_value(obj: np.int32).Should().Be(999999L);
            np.ma.default_fill_value(obj: np.@bool).Should().Be(true);
            np.ma.default_fill_value(obj: np.complex128).Should().Be(new Complex(1e20, 0));
            np.ma.default_fill_value(obj: (DType)null).Should().Be("?");
            np.ma.default_fill_value(obj: np.array(new byte[] { 1 })).Should().Be(999999L);
            typeof(MaskedArrayModule).GetMethods().Where(m => m.Name == "default_fill_value")
                .Select(m => m.GetParameters().Single().Name).Should().AllBe("obj");
        }

        /// <summary>np.ma.size(obj, axis): 6 / 2 / 3 for a (2, 3) masked array, 2 for a list's axis 1, AxisError for a
        ///     scalar's axis 0.</summary>
        [TestMethod]
        public void MaSize_ObjAndAxis_AsNumPy()
        {
            var mm = np.ma.masked_array(A, mask: np.array(new[,] { { false, true, false }, { false, false, true } }));
            np.ma.size(mm).Should().Be(6);
            np.ma.size(obj: mm, axis: 0).Should().Be(2);
            np.ma.size(mm, -1).Should().Be(3);
            np.ma.size(new[,] { { 1, 2 }, { 3, 4 } }, 1).Should().Be(2);
            Action scalar = () => np.ma.size(5, 0);
            scalar.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        /// <summary>MaskedArray.clip(min, max): NumPy 2.x's names and None defaults (one bound, or a copy), the mask kept.</summary>
        [TestMethod]
        public void MaskedArrayClip_MinMax_AsNumPy()
        {
            var mm = np.ma.masked_array(A, mask: np.array(new[,] { { false, true, false }, { false, false, true } }));
            mm.clip(min: 1.0, max: 3.0).filled(-1).flatten().ToArray<double>().Should().Equal(1, -1, 2, 3, 3, -1);
            mm.clip(min: 2.0).filled(-1).flatten().ToArray<double>().Should().Equal(2, -1, 2, 3, 4, -1);
            mm.clip().filled(-1).flatten().ToArray<double>().Should().Equal(0, -1, 2, 3, 4, -1);
            mm.clip(min: 1.0, max: 3.0).mask.flatten().ToArray<bool>().Should().Equal(false, true, false, false, false, true);
        }

        /// <summary>MaskedArray.resize raises NumPy's ValueError with its text (a masked array does not own its data).</summary>
        [TestMethod]
        public void MaskedArrayResize_RaisesNumPysValueError()
        {
            var mm = np.ma.masked_array(A);
            Action resize = () => mm.resize(newshape: new Shape(3, 2));
            resize.Should().Throw<ValueError>().WithMessage(
                "A masked array does not own its data and therefore cannot be resized.\nUse the numpy.ma.resize function instead.");
        }

        #endregion
    }
}
