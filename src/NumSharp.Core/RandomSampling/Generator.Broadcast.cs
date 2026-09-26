using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // =============================================================================================================
        // Array-valued parameters (NumPy's cont / disc broadcast paths, _common.pyx)
        // =============================================================================================================
        //
        // Every overload below follows NumPy's flow for that sampler, in NumPy's order:
        //   1. convert each parameter (RandomParam: PyArray_FROM_OTF with the 'safe' gate; null = the NumPy default for a
        //      defaulted parameter, Python's None for a required one);
        //   2. every converted parameter 0-d -> the scalar path: read each value the way the C code does
        //      (PyFloat_AsDouble / <int64_t>) and delegate to the scalar overload, which checks and draws exactly as
        //      NumPy's scalar branch;
        //   3. otherwise check each constraint over the whole array (check_array_constraint), resolve the output shape
        //      (size, else the broadcast of the parameters; then validate_output_shape), allocate, and draw one value per
        //      output position in C order with that position's parameters (BroadcastWalk, NumPy's MultiIter walk).
        //
        // Draws keep NumPy's stream. Samplers whose draw count does not depend on the parameters (one standard normal /
        // exponential / uniform per value) bulk-fill those variates first and transform them in place — the same
        // expression per element as the C function, over the same variate. The rest call the per-value kernel of the
        // scalar path with a per-value setup (the same statements NumPy runs per call) over a read-ahead buffer whose Owed
        // stays at the values still owed — valid only when EVERY position draws at least once, which a scan over the
        // parameter elements decides (RandomBroadcast.Any); when some position draws nothing (weibull a == 0, poisson
        // lam == 0, ...) the buffer holds one draw (capacity 1 is NumPy's per-call sequence).

        /// <summary>
        ///     Draw samples from a Beta distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="a">Alpha, positive (NaN accepted and sampled). Null is Python's <c>None</c> (NaN on the broadcast
        ///     path; a <see cref="TypeError"/> when every parameter is 0-d).</param>
        /// <param name="b">Beta, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape; a given size
        ///     must be that shape or one the parameters broadcast into.</param>
        /// <returns>The float64 draws; a 0-d array when every parameter is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter's dtype does not cast to float64 under <c>'safe'</c> (complex), or a
        ///     <c>None</c> reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="a"/> (checked first) or <paramref name="b"/> is not
        ///     positive, or the shapes do not broadcast / the size is incompatible (NumPy's texts).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html
        ///     <br/>NumPy's <c>cont_broadcast_2</c> over the modern <c>random_beta</c>, one call per output position with
        ///     that position's <c>(a, b)</c>: byte-identical to <c>default_rng(seed).beta(a_arr, b_arr, size)</c>.
        /// </remarks>
        public NDArray beta(NDArray a, NDArray b, Shape size = default)
        {
            using var pa = RandomParam.Float64(a);
            using var pb = RandomParam.Float64(b);
            if (pa.IsScalar && pb.IsScalar)
                return beta(pa.ScalarChecked("a", ConstraintType.CONS_POSITIVE), pb.ScalarChecked("b", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pa.Array, "a", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pb.Array, "b", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pa.Array, pb.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                // Every beta draws (Johnk's uniforms or the two gammas), so the read-ahead may run the whole output.
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                using var w = new BroadcastWalk(ret, pa.Array, pb.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xa = (double*)w.A, xb = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (a, b) reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double av = xa[j * w.StrideA], bv = xb[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xa, w.StrideA, xb, w.StrideB);
                            var setup = new BetaSetup(av, bv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = Beta(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from an exponential distribution with an array-valued, broadcast scale.
        /// </summary>
        /// <param name="scale">The scale (1/rate); sign bit must be clear (<c>-0.0</c> rejected), NaN propagates. Null is
        ///     NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="scale"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d scale and no size.</returns>
        /// <exception cref="TypeError"><paramref name="scale"/> does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="ValueError">An element is negative (<c>scale &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html
        ///     <br/><c>scale * standard_exponential()</c> per position: the ziggurat exponentials are bulk-filled (one per
        ///     value, whatever the scale) and scaled in place. Byte-identical to NumPy.
        /// </remarks>
        public NDArray exponential(NDArray scale, Shape size = default)
        {
            using var ps = RandomParam.Float64(scale, 1.0);
            if (ps.IsScalar)
                return exponential(ps.Scalar(), size);

            RandomConstraints.CheckArray(ps.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), ps.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardExponential((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, ps.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, s = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = s[j * w.StrideA] * o[j * w.OutStride];
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a uniform distribution over <c>[low, high)</c> with array-valued, broadcast bounds.
        /// </summary>
        /// <param name="low">Lower boundary (inclusive). Null is NumPy's default, 0.</param>
        /// <param name="high">Upper boundary (exclusive). Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the broadcast shape of <c>low</c> and <c>high - low</c>.</param>
        /// <returns>The float64 draws; a 0-d array when both bounds are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A bound does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="OverflowException">An element of <c>high - low</c> is infinite or NaN — NumPy's
        ///     <c>OverflowError('Range exceeds valid bounds')</c> (<c>'high - low range exceeds valid bounds'</c> on the scalar path).</exception>
        /// <exception cref="ValueError">The bounds do not broadcast (NumPy's <c>np.subtract</c> text: <c>operands could not be
        ///     broadcast together with shapes …</c>), an element of <c>high - low</c> has its sign bit set
        ///     (<c>high - low &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
        ///     <br/>NumPy computes <c>np.subtract(high, low)</c> as a whole array first (its broadcast error is the ufunc's),
        ///     then draws <c>low + range * next_double()</c> per position over the broadcast of <c>low</c> and the range.
        ///     The uniforms are bulk-filled (one per value) and transformed in place. Byte-identical to NumPy.
        /// </remarks>
        public NDArray uniform(NDArray low, NDArray high = null, Shape size = default)
        {
            using var pLow = RandomParam.Float64(low, 0.0);
            using var pHigh = RandomParam.Float64(high, 1.0);
            if (pLow.IsScalar && pHigh.IsScalar)
                return uniform(pLow.Scalar(), pHigh.Scalar(), size);

            using var rangeArr = RandomBroadcast.Ufunc(() => np.subtract(pHigh.Array, pLow.Array));
            using var pRange = RandomParam.Float64(rangeArr);
            if (!RandomBroadcast.AllInRange(pRange.Array, -double.MaxValue, double.MaxValue))
                throw new OverflowException("Range exceeds valid bounds");
            RandomConstraints.CheckArray(pRange.Array, "high - low", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pLow.Array, pRange.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    if (ret.size > 0)
                        _bitGenerator.FillDouble((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, pLow.Array, pRange.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, lo = (double*)w.A, rg = (double*)w.B;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = lo[j * w.StrideA] + rg[j * w.StrideB] * o[j * w.OutStride];
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a normal (Gaussian) distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="loc">Mean of the distribution. Null is NumPy's default, 0.</param>
        /// <param name="scale">Standard deviation; sign bit must be clear (<c>-0.0</c> rejected), NaN propagates. Null is
        ///     NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
        ///     <br/><c>loc + scale * standard_normal()</c> per position: the ziggurat normals are bulk-filled (one per value)
        ///     and transformed in place. Byte-identical to NumPy.
        /// </remarks>
        public NDArray normal(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return normal(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pLoc.Array, pScale.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardNormal((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, l = (double*)w.A, s = (double*)w.B;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = l[j * w.StrideA] + s[j * w.StrideB] * o[j * w.OutStride];
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a standard Gamma distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="shape">The shape parameter; sign bit must be clear, NaN propagates. Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="shape"/>'s shape (or
        ///     <paramref name="out"/>'s when given).</param>
        /// <param name="dtype">Native <c>float64</c> (default) or <c>float32</c>.</param>
        /// <param name="out">Optional C-contiguous output of the loop dtype; its shape must be one the parameter broadcasts
        ///     into. Returned when given.</param>
        /// <returns>The draws (float64, or float32 for <paramref name="dtype"/> float32 unless the scalar path returns the
        ///     float64 0-d scalar NumPy returns for <c>size=None</c>).</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is unsupported, <paramref name="out"/> has the wrong dtype,
        ///     the float64 path cannot cast <paramref name="shape"/> under <c>'safe'</c>, or a <c>None</c> / 0-d complex
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is not C-contiguous/writeable or disagrees with the size or
        ///     the parameter's shape, or an element is negative (<c>shape &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_gamma.html
        ///     <br/>NumPy's order: the dtype, then <c>check_output</c>, then the conversion. float64 converts under
        ///     <c>'safe'</c> (<c>cont</c>); float32 under <c>FORCECAST</c> (<c>cont_f</c>): every dtype is cast (complex keeps
        ///     its real part) and the constraint runs on the float32 values, so <c>1e-50</c> becomes a zero shape that draws
        ///     nothing. A zero shape returns 0 without drawing on both paths. Byte-identical to NumPy.
        /// </remarks>
        public NDArray standard_gamma(NDArray shape, Shape size = default, DType dtype = null, NDArray @out = null)
        {
            dtype ??= DType.Double;
            NPTypeCode tc = ResolveFloatDtype(dtype, "standard_gamma");
            if (@out is not null)
                ValidateOut(@out, size, tc, requireCContiguous: true);

            if (tc == NPTypeCode.Single)
            {
                using var pf = RandomParam.Float32Forced(shape);
                if (pf.IsScalar)
                    return standard_gamma(pf.ScalarChecked("shape", ConstraintType.CONS_NON_NEGATIVE), size, dtype, @out);

                RandomConstraints.CheckArray(pf.Array, "shape", ConstraintType.CONS_NON_NEGATIVE);
                var retF = @out ?? RandomBroadcast.NewOutput(NPTypeCode.Single, RandomBroadcast.OutputDims(size, !IsNoSize(size), pf.Array));
                if (@out is not null)
                    RandomBroadcast.OutputDims(@out.Shape, true, pf.Array);
                unsafe
                {
                    using var w = new BroadcastWalk(retF, pf.Array);
                    lock (_bitGenerator.@lock)
                        while (w.Next())
                        {
                            float* o = (float*)w.Out, s = (float*)w.A;
                            for (long j = 0; j < w.Count; j++)
                                o[j * w.OutStride] = NextStandardGammaF(s[j * w.StrideA]);
                        }
                }
                return retF;
            }

            using var p = RandomParam.Float64(shape);
            if (p.IsScalar)
                return standard_gamma(p.ScalarChecked("shape", ConstraintType.CONS_NON_NEGATIVE), size, dtype, @out);

            RandomConstraints.CheckArray(p.Array, "shape", ConstraintType.CONS_NON_NEGATIVE);
            var ret = @out ?? RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            if (@out is not null)
                RandomBroadcast.OutputDims(@out.Shape, true, p.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                // A zero shape draws nothing; any such element forbids the read-ahead (capacity 1 = per draw).
                bool allDraw = !RandomBroadcast.AnyZero(p.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, s = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal shape reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double sv = s[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, s, w.StrideA);
                            var g = new GammaSetup(sv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = StandardGamma(ref src, in g);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Gamma distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="shape">The shape; sign bit must be clear, NaN propagates. Null is Python's <c>None</c>.</param>
        /// <param name="scale">The scale; sign bit must be clear, NaN propagates. Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="shape"/> (checked first) or <paramref name="scale"/>
        ///     is negative, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html
        ///     <br/><c>scale * standard_gamma(shape)</c> per position (a zero shape returns 0 without drawing).
        ///     Byte-identical to NumPy.
        /// </remarks>
        public NDArray gamma(NDArray shape, NDArray scale = null, Shape size = default)
        {
            using var pShape = RandomParam.Float64(shape);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pShape.IsScalar && pScale.IsScalar)
                return gamma(pShape.ScalarChecked("shape", ConstraintType.CONS_NON_NEGATIVE), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pShape.Array, "shape", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pShape.Array, pScale.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                bool allDraw = !RandomBroadcast.AnyZero(pShape.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, pShape.Array, pScale.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, sh = (double*)w.A, sc = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal shape reuses its setup; the scale only multiplies (bit-neutral, see RandomBroadcast.RunEnd).
                            double shv = sh[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, sh, w.StrideA);
                            var g = new GammaSetup(shv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = sc[j * w.StrideB] * StandardGamma(ref src, in g);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from an F distribution with array-valued, broadcast degrees of freedom.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="dfden">Denominator degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="dfnum"/> (checked first) or <paramref name="dfden"/>
        ///     is not positive, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.f.html
        ///     <br/>NumPy's <c>random_f</c> per position, numerator first. Byte-identical to NumPy.
        /// </remarks>
        public NDArray f(NDArray dfnum, NDArray dfden, Shape size = default)
        {
            using var pNum = RandomParam.Float64(dfnum);
            using var pDen = RandomParam.Float64(dfden);
            if (pNum.IsScalar && pDen.IsScalar)
                return f(pNum.ScalarChecked("dfnum", ConstraintType.CONS_POSITIVE), pDen.ScalarChecked("dfden", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pNum.Array, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pDen.Array, "dfden", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pNum.Array, pDen.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                // A position draws unless BOTH chi-squares' df / 2 are 0 (a 5e-324 df halves to 0 and its gamma returns 0
                // without drawing). When either parameter has no such element, every position draws; otherwise read per
                // draw (capacity 1 is NumPy's own call sequence), never ahead of it.
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(pNum.Array) || !RandomBroadcast.AnyHalvesToZero(pDen.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, pNum.Array, pDen.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xn = (double*)w.A, xd = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (dfnum, dfden) reuses both chi-square setups (bit-neutral, see RandomBroadcast.RunEnd).
                            double num = xn[j * w.StrideA], den = xd[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xd, w.StrideB);
                            var halfNum = new GammaSetup(num / 2.0);
                            var halfDen = new GammaSetup(den / 2.0);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = F(ref src, in halfNum, in halfDen, num, den);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the noncentral F distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="dfden">Denominator degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when every parameter is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="dfnum"/> / <paramref name="dfden"/> not positive,
        ///     <paramref name="nonc"/> negative; or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.noncentral_f.html
        ///     <br/>NumPy's <c>random_noncentral_f</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray noncentral_f(NDArray dfnum, NDArray dfden, NDArray nonc, Shape size = default)
        {
            using var pNum = RandomParam.Float64(dfnum);
            using var pDen = RandomParam.Float64(dfden);
            using var pNonc = RandomParam.Float64(nonc);
            if (pNum.IsScalar && pDen.IsScalar && pNonc.IsScalar)
                return noncentral_f(pNum.ScalarChecked("dfnum", ConstraintType.CONS_POSITIVE),
                    pDen.ScalarChecked("dfden", ConstraintType.CONS_POSITIVE),
                    pNonc.ScalarChecked("nonc", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(pNum.Array, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pDen.Array, "dfden", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pNonc.Array, "nonc", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double,
                RandomBroadcast.OutputDims(size, !IsNoSize(size), pNum.Array, pDen.Array, pNonc.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                // A position draws when its denominator chi-square does (dfden / 2 != 0) or its numerator does (see
                // NoncentralChisquareAllDraw). If neither is guaranteed for every position, read per draw.
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(pDen.Array) || NoncentralChisquareAllDraw(pNum.Array, pNonc.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, pNum.Array, pDen.Array, pNonc.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xn = (double*)w.A, xd = (double*)w.B, xc = (double*)w.C;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (dfnum, dfden, nonc) reuses both setups (bit-neutral, see RandomBroadcast.RunEnd).
                            double num = xn[j * w.StrideA], den = xd[j * w.StrideB], nc = xc[j * w.StrideC];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xd, w.StrideB, xc, w.StrideC);
                            var numSetup = new NoncentralChisquareSetup(num, nc);
                            var halfDen = new GammaSetup(den / 2.0);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = NoncentralF(ref src, in numSetup, in halfDen, num, den);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a chi-square distribution with array-valued, broadcast degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="df"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="df"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="df"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>df &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.chisquare.html
        ///     <br/><c>2 * standard_gamma(df / 2)</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray chisquare(NDArray df, Shape size = default)
        {
            using var p = RandomParam.Float64(df);
            if (p.IsScalar)
                return chisquare(p.ScalarChecked("df", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "df", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                // df > 0 (or NaN) draws unless df / 2 underflows to 0 (the gamma of shape 0 returns 0 without drawing).
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(p.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, d = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal df reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double dv = d[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, d, w.StrideA);
                            var half = new GammaSetup(dv / 2.0);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = Chisquare(ref src, in half);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN returns NaN without drawing). Null is
        ///     Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="df"/> (checked first) is not positive or of
        ///     <paramref name="nonc"/> negative, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.noncentral_chisquare.html
        ///     <br/>The modern <c>random_noncentral_chisquare</c> per position; a NaN non-centrality returns NaN without
        ///     drawing (so an array holding one reads per draw). Byte-identical to NumPy.
        /// </remarks>
        public NDArray noncentral_chisquare(NDArray df, NDArray nonc, Shape size = default)
        {
            using var pDf = RandomParam.Float64(df);
            using var pNonc = RandomParam.Float64(nonc);
            if (pDf.IsScalar && pNonc.IsScalar)
                return noncentral_chisquare(pDf.ScalarChecked("df", ConstraintType.CONS_POSITIVE),
                    pNonc.ScalarChecked("nonc", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(pDf.Array, "df", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pNonc.Array, "nonc", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pDf.Array, pNonc.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                bool allDraw = NoncentralChisquareAllDraw(pDf.Array, pNonc.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, pDf.Array, pNonc.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xd = (double*)w.A, xc = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (df, nonc) reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double dv = xd[j * w.StrideA], cv = xc[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xd, w.StrideA, xc, w.StrideB);
                            var setup = new NoncentralChisquareSetup(dv, cv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = NoncentralChisquare(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Whether EVERY broadcast position of the modern <c>random_noncentral_chisquare</c> is guaranteed to draw at
        ///     least once, decided from the parameter elements alone — the condition a read-ahead needs.
        /// </summary>
        /// <param name="df">The converted degrees of freedom (C-contiguous float64).</param>
        /// <param name="nonc">The converted non-centralities (C-contiguous float64).</param>
        /// <returns>True when no position can draw nothing; false when some position might (the caller then reads per draw).</returns>
        /// <remarks>
        ///     <para>
        ///     A position draws nothing exactly when: its <c>nonc</c> is NaN (NaN returned at once); or <c>nonc == 0</c> and
        ///     <c>df / 2 == 0</c> (a chi-square of a shape-0 gamma); or <c>df &lt;= 1</c>, <c>nonc / 2 == 0</c> (a Poisson of
        ///     mean 0 returns 0 without drawing) and <c>df / 2 == 0</c>. The <c>df &gt; 1</c> branch always draws its normal.
        ///     Every non-NaN case therefore needs <c>df / 2 == 0</c> AND <c>nonc / 2 == 0</c> together.
        ///     </para>
        ///     <para>
        ///     The test is conservative over the elements (a pair is only possible if both kinds of element exist), which
        ///     is sound: a false answer only costs the read-ahead, never a draw NumPy does not make.
        ///     </para>
        /// </remarks>
        private static bool NoncentralChisquareAllDraw(NDArray df, NDArray nonc)
            => !RandomBroadcast.AnyNaN(nonc)
               && (!RandomBroadcast.AnyHalvesToZero(df) || !RandomBroadcast.AnyHalvesToZero(nonc));

        /// <summary>
        ///     Draw samples from a standard Student's t distribution with array-valued, broadcast degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="df"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="df"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="df"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>df &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_t.html
        ///     <br/><c>sqrt(df/2) * N / sqrt(G(df/2))</c> per position, the normal drawn first. Byte-identical to NumPy.
        /// </remarks>
        public NDArray standard_t(NDArray df, Shape size = default)
        {
            using var p = RandomParam.Float64(df);
            if (p.IsScalar)
                return standard_t(p.ScalarChecked("df", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "df", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity); // the normal always draws
                using var w = new BroadcastWalk(ret, p.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, d = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal df reuses its gamma setup and sqrt(df / 2) (bit-neutral, see RandomBroadcast.RunEnd).
                            double v = d[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, d, w.StrideA);
                            var half = new GammaSetup(v / 2);
                            var sqrtHalf = Math.Sqrt(v / 2);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = StandardT(ref src, in half, sqrtHalf);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a von Mises distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="mu">Mode ("center"). Null is Python's <c>None</c>.</param>
        /// <param name="kappa">Concentration, non-negative (<c>-0.0</c> rejected; NaN returns NaN without drawing). Null is
        ///     Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws in <c>[-pi, pi]</c>; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="kappa"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.vonmises.html
        ///     <br/>The modern <c>random_vonmises</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray vonmises(NDArray mu, NDArray kappa, Shape size = default)
        {
            using var pMu = RandomParam.Float64(mu);
            using var pKappa = RandomParam.Float64(kappa);
            if (pMu.IsScalar && pKappa.IsScalar)
                return vonmises(pMu.Scalar(), pKappa.ScalarChecked("kappa", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(pKappa.Array, "kappa", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pMu.Array, pKappa.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                bool allDraw = !RandomBroadcast.AnyNaN(pKappa.Array);
                var src = new DrawBuffer64(_bitGenerator, storage, allDraw ? DrawBuffer64.Capacity : 1);
                using var w = new BroadcastWalk(ret, pMu.Array, pKappa.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xm = (double*)w.A, xk = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (mu, kappa) reuses its envelope (bit-neutral, see RandomBroadcast.RunEnd).
                            double mv = xm[j * w.StrideA], kv = xk[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xm, w.StrideA, xk, w.StrideB);
                            var setup = new VonmisesSetup(mv, kv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = Vonmises(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Pareto II (Lomax) distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="a">Shape, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>a &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.pareto.html
        ///     <br/><c>expm1(E / a)</c> per position over bulk-filled exponentials; the values carry the scalar
        ///     <see cref="pareto(double, Shape)"/>'s documented <c>expm1</c> band difference (see <see cref="Expm1"/>).
        /// </remarks>
        public NDArray pareto(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return pareto(p.ScalarChecked("a", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardExponential((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = Expm1(o[j * w.OutStride] / x[j * w.StrideA]);
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Weibull distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="a">Shape, non-negative (<c>-0.0</c> rejected; NaN accepted); 0 returns 0 without drawing. Null is
        ///     Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is negative (<c>a &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.weibull.html
        ///     <br/><c>E^(1/a)</c> per position, 0 without a draw where <c>a == 0</c>. The exponentials the drawing positions
        ///     take are bulk-filled into the TAIL of the output and consumed front to back — a position never reads a slot
        ///     an earlier position overwrote, because at most as many positions precede it as draw-free positions exist.
        ///     Byte-identical to NumPy.
        /// </remarks>
        public NDArray weibull(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return weibull(p.ScalarChecked("a", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                long n = ret.size;
                var dst = (double*)ret.Address;
                // How many positions draw: all of them unless an element is 0, else counted over the walk.
                long draws = n;
                if (RandomBroadcast.AnyZero(p.Array))
                {
                    draws = 0;
                    using var count = new BroadcastWalk(ret, p.Array);
                    while (count.Next())
                    {
                        double* x = (double*)count.A;
                        for (long j = 0; j < count.Count; j++)
                            if (x[j * count.StrideA] != 0.0)
                                draws++;
                    }
                }
                lock (_bitGenerator.@lock)
                    FillStandardExponential(dst + (n - draws), draws);
                long next = n - draws;
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                    {
                        double av = x[j * w.StrideA];
                        // random_weibull: a == 0 returns 0 before drawing; else pow(E, 1. / a).
                        o[j * w.OutStride] = av == 0.0 ? 0.0 : Math.Pow(dst[next++], 1.0 / av);
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples in <c>[0, 1]</c> from a power distribution with an array-valued, broadcast exponent.
        /// </summary>
        /// <param name="a">Parameter, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>a &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.power.html
        ///     <br/><c>(-expm1(-E))^(1/a)</c> per position over bulk-filled exponentials; the values carry the scalar
        ///     <see cref="power(double, Shape)"/>'s documented <c>expm1</c> band difference.
        /// </remarks>
        public NDArray power(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return power(p.ScalarChecked("a", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardExponential((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = Math.Pow(-Expm1(-o[j * w.OutStride]), 1.0 / x[j * w.StrideA]);
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the Laplace distribution with array-valued, broadcast location and scale.
        /// </summary>
        /// <param name="loc">The peak's position. Null is NumPy's default, 0.</param>
        /// <param name="scale">The exponential decay, non-negative (<c>-0.0</c> rejected; NaN accepted). Null is NumPy's
        ///     default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.laplace.html
        ///     <br/><c>random_laplace</c> per position (one uniform, redrawn only on an exact 0). Byte-identical to NumPy.
        /// </remarks>
        public NDArray laplace(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return laplace(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pLoc.Array, pScale.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, l = (double*)w.A, s = (double*)w.B;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = n - i;
                            o[j * w.OutStride] = Distributions.RandomLaplace(ref src, l[j * w.StrideA], s[j * w.StrideB]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Gumbel distribution with array-valued, broadcast location and scale.
        /// </summary>
        /// <param name="loc">The mode's location. Null is NumPy's default, 0.</param>
        /// <param name="scale">The scale, non-negative (<c>-0.0</c> rejected; NaN accepted). Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gumbel.html
        ///     <br/><c>random_gumbel</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray gumbel(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return gumbel(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pLoc.Array, pScale.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, l = (double*)w.A, s = (double*)w.B;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = n - i;
                            o[j * w.OutStride] = Distributions.RandomGumbel(ref src, l[j * w.StrideA], s[j * w.StrideB]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a logistic distribution with array-valued, broadcast location and scale.
        /// </summary>
        /// <param name="loc">The mean. Null is NumPy's default, 0.</param>
        /// <param name="scale">The scale, non-negative (<c>-0.0</c> rejected; NaN accepted). Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.logistic.html
        ///     <br/><c>random_logistic</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray logistic(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return logistic(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pLoc.Array, pScale.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, l = (double*)w.A, s = (double*)w.B;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = n - i;
                            o[j * w.OutStride] = Distributions.RandomLogistic(ref src, l[j * w.StrideA], s[j * w.StrideB]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a log-normal distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="mean">Mean of the underlying normal. Null is NumPy's default, 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal, non-negative (<c>-0.0</c> rejected; NaN
        ///     accepted). Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="sigma"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.lognormal.html
        ///     <br/><c>exp(mean + sigma * N)</c> per position over bulk-filled ziggurat normals. Byte-identical to NumPy.
        /// </remarks>
        public NDArray lognormal(NDArray mean, NDArray sigma = null, Shape size = default)
        {
            using var pMean = RandomParam.Float64(mean, 0.0);
            using var pSigma = RandomParam.Float64(sigma, 1.0);
            if (pMean.IsScalar && pSigma.IsScalar)
                return lognormal(pMean.Scalar(), pSigma.Scalar(), size);

            RandomConstraints.CheckArray(pSigma.Array, "sigma", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pMean.Array, pSigma.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardNormal((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, pMean.Array, pSigma.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, m = (double*)w.A, s = (double*)w.B;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = Math.Exp(m[j * w.StrideA] + s[j * w.StrideB] * o[j * w.OutStride]);
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Rayleigh distribution with an array-valued, broadcast scale.
        /// </summary>
        /// <param name="scale">Scale (also the mode), non-negative (<c>-0.0</c> rejected; NaN accepted). Null is NumPy's
        ///     default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="scale"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d scale and no size.</returns>
        /// <exception cref="TypeError"><paramref name="scale"/> does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element is negative (<c>scale &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.rayleigh.html
        ///     <br/>The modern <c>mode * sqrt(2 * E)</c> per position over bulk-filled exponentials. Byte-identical to NumPy.
        /// </remarks>
        public NDArray rayleigh(NDArray scale, Shape size = default)
        {
            using var p = RandomParam.Float64(scale, 1.0);
            if (p.IsScalar)
                return rayleigh(p.Scalar(), size);

            RandomConstraints.CheckArray(p.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                lock (_bitGenerator.@lock)
                    FillStandardExponential((double*)ret.Address, ret.size);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, s = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = s[j * w.StrideA] * Math.Sqrt(2.0 * o[j * w.OutStride]);
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Wald (inverse Gaussian) distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="mean">Distribution mean, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="scale">Scale, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="mean"/> (checked first) or <paramref name="scale"/> is
        ///     not positive, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.wald.html
        ///     <br/>The modern <c>random_wald</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray wald(NDArray mean, NDArray scale, Shape size = default)
        {
            using var pMean = RandomParam.Float64(mean);
            using var pScale = RandomParam.Float64(scale);
            if (pMean.IsScalar && pScale.IsScalar)
                return wald(pMean.ScalarChecked("mean", ConstraintType.CONS_POSITIVE), pScale.ScalarChecked("scale", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pMean.Array, "mean", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, RandomBroadcast.OutputDims(size, !IsNoSize(size), pMean.Array, pScale.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity); // a normal and a uniform per value
                using var w = new BroadcastWalk(ret, pMean.Array, pScale.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, m = (double*)w.A, s = (double*)w.B;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = n - i;
                            o[j * w.OutStride] = Wald(ref src, m[j * w.StrideA], s[j * w.StrideB]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the triangular distribution over <c>[left, right]</c> with array-valued, broadcast limits.
        /// </summary>
        /// <param name="left">Lower limit. Null is Python's <c>None</c>.</param>
        /// <param name="mode">The peak, <c>left &lt;= mode &lt;= right</c>. Null is Python's <c>None</c>.</param>
        /// <param name="right">Upper limit, greater than <paramref name="left"/>. Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the limits' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when every limit is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A limit does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches the
        ///     scalar path (read in NumPy's order: left, right, mode).</exception>
        /// <exception cref="ValueError">In NumPy's order: any <c>left &gt; mode</c>, any <c>mode &gt; right</c>, any
        ///     <c>left == right</c>; the comparisons' own broadcast errors (<c>operands could not be broadcast together with
        ///     shapes …</c>); or an incompatible size.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.triangular.html
        ///     <br/>NumPy's three whole-array comparisons (<c>np.greater</c> / <c>np.equal</c>), then
        ///     <c>cont_broadcast_3</c> over <c>random_triangular</c>. A NaN limit passes every comparison and samples NaN.
        ///     Byte-identical to NumPy.
        /// </remarks>
        public NDArray triangular(NDArray left, NDArray mode, NDArray right, Shape size = default)
        {
            using var pLeft = RandomParam.Float64(left);
            using var pMode = RandomParam.Float64(mode);
            using var pRight = RandomParam.Float64(right);
            if (pLeft.IsScalar && pMode.IsScalar && pRight.IsScalar)
            {
                // NumPy reads fleft, fright, fmode — in that order — before any comparison.
                double fleft = pLeft.Scalar(), fright = pRight.Scalar(), fmode = pMode.Scalar();
                return triangular(fleft, fmode, fright, size);
            }

            if (RandomBroadcast.AnyCompare(NDExpr.Greater, pLeft.Array, pMode.Array))
                throw new ValueError("left > mode");
            if (RandomBroadcast.AnyCompare(NDExpr.Greater, pMode.Array, pRight.Array))
                throw new ValueError("mode > right");
            if (RandomBroadcast.AnyCompare(NDExpr.Equal, pLeft.Array, pRight.Array))
                throw new ValueError("left == right");
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double,
                RandomBroadcast.OutputDims(size, !IsNoSize(size), pLeft.Array, pMode.Array, pRight.Array));
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pLeft.Array, pMode.Array, pRight.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, l = (double*)w.A, m = (double*)w.B, r = (double*)w.C;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = n - i;
                            o[j * w.OutStride] = Distributions.RandomTriangular(ref src, l[j * w.StrideA], m[j * w.StrideB], r[j * w.StrideC]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a binomial distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="n">Number of trials, a non-negative INTEGER array (a float array is NumPy's safe-cast
        ///     <see cref="TypeError"/>). Null is Python's <c>None</c> (a <see cref="TypeError"/>).</param>
        /// <param name="p">Probability of success, in <c>[0, 1]</c> (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The int64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> (converted first) does not cast to float64 or
        ///     <paramref name="n"/> to int64 under <c>'safe'</c>, or a <c>None</c> reaches a conversion / the scalar path.</exception>
        /// <exception cref="ValueError">In NumPy's order: an element of <paramref name="p"/> outside <c>[0, 1]</c> or NaN,
        ///     then of <paramref name="n"/> negative; or the shapes do not broadcast / the size is incompatible (NumPy numbers
        ///     <paramref name="p"/> before <paramref name="n"/> in the mismatch text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.binomial.html
        ///     <br/>NumPy's own broadcast loop over the modern <c>random_binomial</c> with this Generator's setup cache; an
        ///     <c>n == 0</c> or <c>p == 0</c> position returns 0 without drawing. Byte-identical to NumPy.
        /// </remarks>
        public NDArray binomial(NDArray n, NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            using var pn = RandomParam.Int64(n);
            if (pp.IsScalar && pn.IsScalar)
                return binomial(pn.ScalarInt64(), pp.Scalar(), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_0_1);
            RandomConstraints.CheckArray(pn.Array, "n", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), pp.Array, pn.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = !RandomBroadcast.AnyZero(pn.Array) && !RandomBroadcast.AnyZero(pp.Array);
                var src = new DrawBufferDouble(_bitGenerator, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pp.Array, pn.Array);
                lock (_bitGenerator.@lock)
                {
                    // Keys that recur NON-consecutively (a count array cycling through values) thrash NumPy's single
                    // binomial cache; the per-fill memo keeps each recurring key's setup (and its memoized inversion walk)
                    // for the fill — bit-neutral (BinomialState), a key memoized only from its second miss, so positions
                    // that never repeat a key allocate nothing — and is removed before the lock is released, leaving the
                    // last key's setup current exactly as NumPy's single entry.
                    if (count >= 8)
                        _binomial.Memo = new BinomialSetup[1 << BinomialState.MemoBits];
                    try
                    {
                        while (w.Next())
                        {
                            long* o = (long*)w.Out, xn = (long*)w.B;
                            double* xp = (double*)w.A;
                            for (long j = 0; j < w.Count; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = Distributions.RandomBinomial(ref src, xp[j * w.StrideA], xn[j * w.StrideB], _binomial);
                            }
                        }
                    }
                    finally
                    {
                        _binomial.Memo = null;
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a negative binomial distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="n">Parameter, positive and not NaN (need not be an integer). Null is Python's <c>None</c> — NumPy
        ///     reads it as NaN here (<c>n must not be NaN</c>), not as a <see cref="TypeError"/>.</param>
        /// <param name="p">Probability of success, in <c>(0, 1]</c>. Null is Python's <c>None</c> (NaN, rejected).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The int64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> (converted first) or <paramref name="n"/> does not cast to
        ///     float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="n"/> NaN or not positive, <paramref name="p"/>
        ///     outside <c>(0, 1]</c>, the implied Poisson mean <c>(1-p)/p * (n + 10*sqrt(n))</c> above the int64-safe bound
        ///     (<c>n too large or p too small, see Generator.negative_binomial Notes</c>) — whose whole-array arithmetic
        ///     raises NumPy's ufunc broadcast error for mismatched shapes — or an incompatible size.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.negative_binomial.html
        ///     <br/>The modern <c>random_negative_binomial</c> per position: a Poisson of a gamma draw. Byte-identical to NumPy.
        /// </remarks>
        public NDArray negative_binomial(NDArray n, NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            using var pn = RandomParam.Float64(n);
            if (pp.IsScalar && pn.IsScalar)
                // NumPy's scalar branch reads PyArray_DATA of the converted arrays (a None is already NaN), not the objects.
                return negative_binomial(pn.Array.GetAtIndex<double>(0), pp.Array.GetAtIndex<double>(0), size);

            RandomConstraints.CheckArray(pn.Array, "n", ConstraintType.CONS_POSITIVE_NOT_NAN);
            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_GT_0_1);
            if (NegativeBinomialLamTooLarge(pn.Array, pp.Array))
                throw new ValueError("n too large or p too small, see Generator.negative_binomial Notes");
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), pn.Array, pp.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity); // the gamma (n > 0) always draws
                using var w = new BroadcastWalk(ret, pn.Array, pp.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* xn = (double*)w.A, xp = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (n, p) reuses the gamma setup and its scale (bit-neutral, see RandomBroadcast.RunEnd).
                            double nv = xn[j * w.StrideA], pv = xp[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xp, w.StrideB);
                            var g = new GammaSetup(nv);
                            var scale = (1 - pv) / pv;
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = NegativeBinomial(ref src, in g, scale);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     NumPy's whole-array bound of the negative binomial's Poisson mean:
        ///     <c>np.any((1 - p) / p * (n + 10 * np.sqrt(n)) &gt; POISSON_LAM_MAX)</c>.
        /// </summary>
        /// <param name="n">The converted <c>n</c>.</param>
        /// <param name="p">The converted <c>p</c>.</param>
        /// <returns>True when some broadcast pair exceeds the bound.</returns>
        /// <exception cref="ValueError"><paramref name="p"/> and <paramref name="n"/> do not broadcast — the ufunc's text,
        ///     which names <paramref name="p"/>'s shape first as NumPy's product does.</exception>
        /// <remarks>Each operation is NumPy's elementwise IEEE operation in NumPy's order, so the bound is decided on the same bits.</remarks>
        private static bool NegativeBinomialLamTooLarge(NDArray n, NDArray p)
        {
            using var oneMinusP = 1.0 - p;
            using var ratio = oneMinusP / p;
            using var sqrtN = np.sqrt(n);
            using var tenSqrt = 10.0 * sqrtN;
            using var shifted = n + tenSqrt;
            using var maxLam = RandomBroadcast.Ufunc(() => ratio * shifted);
            return RandomBroadcast.AnyInRange(maxLam, Math.BitIncrement(PoissonLamMax), double.PositiveInfinity);
        }

        /// <summary>
        ///     Draw samples from a Poisson distribution with an array-valued, broadcast mean.
        /// </summary>
        /// <param name="lam">Expected number of events, non-negative, not NaN, at most <c>int64 max - 10*sqrt(int64
        ///     max)</c>. Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="lam"/>'s shape.</param>
        /// <returns>The int64 draws; a 0-d array for a 0-d mean and no size.</returns>
        /// <exception cref="TypeError"><paramref name="lam"/> does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element is too large or NaN (<c>lam value too large</c> — the array test runs the
        ///     bound FIRST, so NaN reports it) or negative (<c>lam &lt; 0 or lam contains NaNs</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.poisson.html
        ///     <br/><c>random_poisson</c> per position (product method below 10, PTRS above, 0 without a draw for
        ///     <c>lam == 0</c>). Byte-identical to NumPy.
        /// </remarks>
        public NDArray poisson(NDArray lam, Shape size = default)
        {
            using var p = RandomParam.Float64(lam, 1.0);
            if (p.IsScalar)
                return poisson(p.Scalar(), size);

            RandomConstraints.CheckArray(p.Array, "lam", ConstraintType.CONS_POISSON);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = !RandomBroadcast.AnyZero(p.Array);
                var src = new DrawBufferDouble(_bitGenerator, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* l = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal means shares NumPy's per-call statements for the mean (PTRS constants or
                            // exp(-lam)), built once and read back (bit-neutral: RandomBroadcast.RunEnd, PoissonSetup), and
                            // a long run hoists PTRS's per-rejection log(invalpha) too. A short run keeps them in locals as
                            // NumPy's call does: a PoissonSetup per value costs more than the exp it holds.
                            double lv = l[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, l, w.StrideA);
                            if (end - j >= RandomBroadcast.PoissonSetupMinRun)
                            {
                                var setup = new PoissonSetup(lv, hoistRejectionLog: end - j >= RandomBroadcast.PoissonRejectionLogMinRun);
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = Distributions.RandomPoisson(ref src, in setup);
                                }
                            }
                            else
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = Distributions.RandomPoisson(ref src, lv);
                                }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Zipf distribution with an array-valued, broadcast parameter.
        /// </summary>
        /// <param name="a">Distribution parameter, greater than 1 (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="a"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>); a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not above 1 or is NaN (<c>a &lt;= 1 or a contains NaNs</c>), or the
        ///     size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.zipf.html
        ///     <br/>The modern <c>random_zipf</c> per position; <c>a &gt;= 1025</c> (infinity included) returns 1 without
        ///     drawing. Byte-identical to NumPy.
        /// </remarks>
        public NDArray zipf(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return zipf(p.ScalarChecked("a", ConstraintType.CONS_GT_1), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_GT_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), p.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = !RandomBroadcast.AnyInRange(p.Array, 1025.0, double.PositiveInfinity); // ZipfSetup.Degenerate: 1 without a draw
                var src = new DrawBufferDouble(_bitGenerator, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal exponents reuses its setup and — when as long as the scalar fill's threshold —
                            // the acceptance test's pow(1 + 1/X, a - 1) memo for the small candidates, which belongs to one
                            // exponent (both bit-neutral, see RandomBroadcast.RunEnd and Zipf).
                            double av = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            var setup = new ZipfSetup(av);
                            double[] tMemo = null;
                            if (end - j >= 16 && !setup.Degenerate)
                            {
                                tMemo = new double[256];
                                Array.Fill(tMemo, double.NaN);
                            }
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = Zipf(ref src, in setup, tMemo);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from the geometric distribution with an array-valued, broadcast probability.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial, in <c>(0, 1]</c>. Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="p"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>); a 0-d array for a 0-d <paramref name="p"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is outside <c>(0, 1]</c> or NaN, or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.geometric.html
        ///     <br/>The modern <c>random_geometric</c> per position: the CDF search (one uniform) for <c>p &gt;= 1/3</c>,
        ///     else the inversion over a ziggurat exponential. Both take their draws from one 64-bit read-ahead — a uniform
        ///     is one 64-bit unit — and every value draws, so the read-ahead serves any mix. Byte-identical to NumPy.
        /// </remarks>
        public NDArray geometric(NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            if (pp.IsScalar)
                return geometric(pp.ScalarChecked("p", ConstraintType.CONS_BOUNDED_GT_0_1), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_GT_0_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), pp.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
                var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);
                using var w = new BroadcastWalk(ret, pp.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal p takes one branch of NumPy's per-call choice and, for the inversion, one
                            // log1p(-p) (bit-neutral, see RandomBroadcast.RunEnd).
                            double pv = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            if (pv >= 0.333333333333333333333333)
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = GeometricSearch(ref src, pv);
                                }
                            else
                            {
                                double log1pNegP = Log1p(-pv); // the inversion's log1p(-p)
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = GeometricInversion(ref src, log1pNegP);
                                }
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     NumPy's <c>random_geometric_search</c> over a <see cref="DrawBuffer64"/> — the twin of
        ///     <see cref="Distributions.RandomGeometricSearch"/> for fills that mix it with the ziggurat inversion.
        /// </summary>
        /// <param name="src">The draw source (its <see cref="DrawBuffer64.NextDouble"/> is the generator's <c>next_double</c>).</param>
        /// <param name="p">The success probability (<c>&gt;= 1/3</c>).</param>
        /// <returns>The trial on which the first success occurs (<c>&gt;= 1</c>).</returns>
        private static long GeometricSearch(ref DrawBuffer64 src, double p)
        {
            long X = 1;
            double sum, prod, q;
            sum = prod = p;
            q = 1.0 - p;
            double U = src.NextDouble();
            while (U > sum)
            {
                prod *= q;
                sum += prod;
                X++;
            }
            return X;
        }

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="ngood">Good selections, a non-negative INTEGER array below <c>10^9</c>. Null is Python's <c>None</c>
        ///     (a <see cref="TypeError"/>).</param>
        /// <param name="nbad">Bad selections, a non-negative integer array below <c>10^9</c>.</param>
        /// <param name="nsample">Items sampled, <c>0 &lt;= nsample &lt;= ngood + nbad</c> per position.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape.</param>
        /// <returns>The int64 draws; a 0-d array when every parameter is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to int64 under <c>'safe'</c> (floats, uint64, complex) or is
        ///     <c>None</c>.</exception>
        /// <exception cref="ValueError">In NumPy's order: an element of <paramref name="ngood"/> or <paramref name="nbad"/>
        ///     at least <c>10^9</c>; any <c>ngood + nbad &lt; nsample</c> (whose whole-array arithmetic raises the ufunc
        ///     broadcast error for mismatched shapes); then a negative <paramref name="ngood"/>, <paramref name="nbad"/> or
        ///     <paramref name="nsample"/>; or an incompatible size.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.hypergeometric.html
        ///     <br/>The modern <c>random_hypergeometric</c> per position: HRUA (doubles) for
        ///     <c>10 &lt;= nsample &lt;= ngood + nbad - 10</c>, else the urn walk (32-bit words). A pre-walk classifies the
        ///     positions: all HRUA reads doubles ahead, all drawing urns read words ahead, and any mix — whose 64-bit and
        ///     buffered 32-bit draws interleave — or a draw-free urn position runs per draw. Byte-identical to NumPy.
        /// </remarks>
        public NDArray hypergeometric(NDArray ngood, NDArray nbad, NDArray nsample, Shape size = default)
        {
            const long HypergeomMax = 1000000000;
            using var pg = RandomParam.Int64(ngood);
            using var pb = RandomParam.Int64(nbad);
            using var ps = RandomParam.Int64(nsample);
            if (pg.IsScalar && pb.IsScalar && ps.IsScalar)
                return hypergeometric(pg.ScalarInt64(), pb.ScalarInt64(), ps.ScalarInt64(), size);

            if (RandomBroadcast.AnyInRange(pg.Array, HypergeomMax, double.PositiveInfinity) || RandomBroadcast.AnyInRange(pb.Array, HypergeomMax, double.PositiveInfinity))
                throw new ValueError($"both ngood and nbad must be less than {HypergeomMax}");
            using (var total = RandomBroadcast.Ufunc(() => np.add(pg.Array, pb.Array)))
                if (RandomBroadcast.AnyAndDispose(RandomBroadcast.Ufunc(() => np.less(total, ps.Array))))
                    throw new ValueError("ngood + nbad < nsample");
            RandomConstraints.CheckArray(pg.Array, "ngood", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.CheckArray(pb.Array, "nbad", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.CheckArray(ps.Array, "nsample", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64,
                RandomBroadcast.OutputDims(size, !IsNoSize(size), pg.Array, pb.Array, ps.Array));
            unsafe
            {
                long count = ret.size;
                // Classify the positions: how many run HRUA, and how many run an urn walk that draws at least one word.
                long hrua = 0, urnDraws = 0;
                using (var scan = new BroadcastWalk(ret, pg.Array, pb.Array, ps.Array))
                    while (scan.Next())
                    {
                        long* g = (long*)scan.A, b = (long*)scan.B, s = (long*)scan.C;
                        for (long j = 0; j < scan.Count; j++)
                        {
                            long good = g[j * scan.StrideA], bad = b[j * scan.StrideB], sample = s[j * scan.StrideC];
                            if (sample >= 10 && sample <= good + bad - 10)
                                hrua++;
                            else if (UrnDraws(good, bad, sample))
                                urnDraws++;
                        }
                    }

                using var w = new BroadcastWalk(ret, pg.Array, pb.Array, ps.Array);
                lock (_bitGenerator.@lock)
                {
                    if (hrua == count || urnDraws == count)
                    {
                        // One algorithm everywhere, every position drawing: a read-ahead of its draw width.
                        double* dStore = stackalloc double[DrawBufferDouble.Capacity];
                        uint* wStore = stackalloc uint[DrawBuffer32.Capacity];
                        var srcD = new DrawBufferDouble(_bitGenerator, dStore, DrawBufferDouble.Capacity);
                        var src32 = new DrawBuffer32(_bitGenerator, wStore, DrawBuffer32.Capacity);
                        bool allHrua = hrua == count;
                        long i = 0;
                        while (w.Next())
                        {
                            long* o = (long*)w.Out, g = (long*)w.A, b = (long*)w.B, s = (long*)w.C;
                            if (allHrua)
                                for (long j = 0; j < w.Count;)
                                {
                                    // HRUA's setup once per run of equal (ngood, nbad, nsample), and — for a run long
                                    // enough to repay it (NewGpMemo's threshold, the scalar fill's) — the per-candidate
                                    // logfactorial memo; both bit-neutral (see RandomBroadcast.RunEnd).
                                    long end = RandomBroadcast.RunEnd(j, w.Count, g, w.StrideA, b, w.StrideB, s, w.StrideC);
                                    var hruaSetup = new HruaSetup(g[j * w.StrideA], b[j * w.StrideB], s[j * w.StrideC]);
                                    double[] gpMemo = hruaSetup.NewGpMemo(end - j);
                                    for (; j < end; j++, i++)
                                    {
                                        srcD.Owed = count - i;
                                        o[j * w.OutStride] = HypergeometricHrua(ref srcD, in hruaSetup, gpMemo);
                                    }
                                }
                            else
                                // Every position walks an urn that draws: no per-call setup to share, one draw loop.
                                for (long j = 0; j < w.Count; j++, i++)
                                {
                                    src32.Owed = count - i;
                                    o[j * w.OutStride] = HypergeometricSample(ref src32, g[j * w.StrideA], b[j * w.StrideB], s[j * w.StrideC]);
                                }
                        }
                    }
                    else
                    {
                        // Mixed widths (or a draw-free urn): per draw, straight from the generator (NumPy's call sequence).
                        double word;
                        var one = new DrawBufferDouble(_bitGenerator, &word, 1);
                        while (w.Next())
                        {
                            long* o = (long*)w.Out, g = (long*)w.A, b = (long*)w.B, s = (long*)w.C;
                            for (long j = 0; j < w.Count; j++)
                                o[j * w.OutStride] = Hypergeometric(ref one, _bitGenerator, g[j * w.StrideA], b[j * w.StrideB], s[j * w.StrideC]);
                        }
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Whether NumPy's urn walk (<c>hypergeometric_sample</c>) draws at least one word for these parameters: after
        ///     its complement it loops while the sample, the good count and the bad count are all positive, and its first
        ///     iteration always draws.
        /// </summary>
        /// <param name="good">Good items.</param>
        /// <param name="bad">Bad items.</param>
        /// <param name="sample">Items sampled.</param>
        /// <returns>True when the walk draws.</returns>
        private static bool UrnDraws(long good, long bad, long sample)
        {
            long total = good + bad;
            long computedSample = sample > total / 2 ? total - sample : sample;
            return computedSample > 0 && good > 0 && bad > 0;
        }

        /// <summary>
        ///     Draw samples from a logarithmic series distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="p">Shape, in <c>[0, 1)</c> (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is <paramref name="p"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>); a 0-d array for a 0-d <paramref name="p"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is outside <c>[0, 1)</c> or NaN (<c>p &lt; 0, p &gt;= 1 or p contains
        ///     NaNs</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.logseries.html
        ///     <br/>The modern <c>random_logseries</c> per position; like the scalar sampler, a count can differ only where
        ///     Windows NumPy's <c>expm1</c> and <see cref="Expm1"/> part by an ulp at a boundary.
        /// </remarks>
        public NDArray logseries(NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            if (pp.IsScalar)
                return logseries(pp.ScalarChecked("p", ConstraintType.CONS_BOUNDED_LT_0_1), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_LT_0_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, RandomBroadcast.OutputDims(size, !IsNoSize(size), pp.Array));
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity); // V is always drawn
                using var w = new BroadcastWalk(ret, pp.Array);
                lock (_bitGenerator.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal p reuses r = log1p(-p) (bit-neutral, see RandomBroadcast.RunEnd).
                            double pv = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            var r = Log1p(-pv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = Logseries(ref src, pv, r);
                            }
                        }
                    }
            }
            return ret;
        }
    }
}
