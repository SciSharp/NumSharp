using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        // =============================================================================================================
        // Array-valued parameters of the legacy RandomState samplers (mtrand.pyx over _common.pyx's cont / disc)
        // =============================================================================================================
        //
        // Every overload below follows mtrand.pyx's flow for that sampler, in its order:
        //   1. convert each parameter (RandomParam: PyArray_FROM_OTF with the 'safe' gate; null = the NumPy default for a
        //      defaulted parameter, Python's None for a required one) — in the order mtrand converts them;
        //   2. every converted parameter 0-d -> the scalar path: read each value the way the C code does
        //      (PyFloat_AsDouble / <int64_t>) and delegate to the scalar overload, which checks and draws exactly as
        //      mtrand's scalar branch;
        //   3. otherwise check each constraint over the whole array (check_array_constraint), allocate the output in
        //      NumPy's order (RandomBroadcast.NewOutput: np.empty(size) BEFORE the parameters are broadcast against it
        //      when a size is given, so an unallocatable size reports "array is too big" ahead of any shape mismatch; the
        //      parameters' broadcast first otherwise; then validate_output_shape), and draw one value per output position
        //      in C order with that position's parameters (BroadcastWalk, NumPy's MultiIter walk).
        //
        // The legacy constraints are mtrand's, not the Generator's: no 'high - low' sign check in uniform, no 10^9 caps
        // and CONS_GTE_1 on nsample in hypergeometric, no Poisson-mean bound in negative_binomial, LEGACY_CONS_POISSON,
        // and LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG for the counts NumPy stores in a C long.
        //
        // `size` follows NumPy exactly here: default is size=None, Shape.Scalar is size=() — a GIVEN 0-d size, which array
        // parameters must broadcast into (NumPy's "Output size () is not compatible with broadcast dimensions of inputs
        // (3,)."). The scalar overloads read Shape.Scalar as None (IsScalarDraw), a convention that is unobservable there
        // (NumPy's size=() and size=None both yield one 0-d draw); the all-0-d path delegates to them with `size`
        // unchanged and so returns that same 0-d draw.
        //
        // Draws keep NumPy's stream, including RandomState's cached Gaussian (legacy_gauss). Samplers whose draw count does
        // not depend on the parameters (one uniform / one legacy_gauss per value) bulk-fill those variates first and
        // transform them in place — the same expression per element as the C function over the same variate. The rest
        // call the per-value legacy kernel of the scalar path with the setup NumPy's call computes (the same statements),
        // built once per RUN of equal parameters (RandomBroadcast.RunEnd — bit-neutral, a setup is a pure function of its
        // parameters), over a read-ahead whose Owed stays at the values still owed — valid only when EVERY position draws
        // at least once, which a scan over the parameter elements decides (RandomBroadcast.AnyZero / AnyNaN / AnyInRange
        // / AnyHalvesToZero); when some position might draw nothing (standard_gamma shape == 0, poisson lam == 0, a
        // degrees of freedom halving to 0, ...) the buffer holds one draw (capacity 1 is NumPy's per-call sequence).
        //
        // Integer outputs are int64: NumPy's legacy samplers return C long, which NumSharp models as LP64's 64-bit long
        // (Linux/macOS NumPy); Windows NumPy's 32-bit long truncates the same values and bounds counts at 2^31 - 1.

        /// <summary>
        ///     Draw samples from a Beta distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="a">Alpha, positive (NaN accepted and sampled). Null is Python's <c>None</c> (NaN on the broadcast
        ///     path; a <see cref="TypeError"/> when every parameter is 0-d).</param>
        /// <param name="b">Beta, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the parameters' broadcast shape; a given size
        ///     (<see cref="Shape.Scalar"/> is NumPy's <c>size=()</c>) must be that shape or one the parameters broadcast
        ///     into.</param>
        /// <returns>The float64 draws; a 0-d array when every parameter is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter's dtype does not cast to float64 under <c>'safe'</c> (complex), or a
        ///     <c>None</c> reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="a"/> (checked first) or <paramref name="b"/> is not
        ///     positive, or the shapes do not broadcast / the size is incompatible (NumPy's texts).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>mtrand's <c>cont(&amp;legacy_beta, …)</c> broadcast path, one <c>legacy_beta</c> call per output position
        ///     with that position's <c>(a, b)</c>: byte-identical to <c>RandomState(seed).beta(a_arr, b_arr, size)</c>.
        /// </remarks>
        public NDArray beta(NDArray a, NDArray b, Shape size = default)
        {
            using var pa = RandomParam.Float64(a);
            using var pb = RandomParam.Float64(b);
            if (pa.IsScalar && pb.IsScalar)
                return beta(pa.ScalarChecked("a", ConstraintType.CONS_POSITIVE), pb.ScalarChecked("b", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pa.Array, "a", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pb.Array, "b", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pa.Array, pb.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // Every legacy_beta draws: Johnk's two uniforms, or two gammas of positive (or NaN) shape — each draws a
                // uniform even when its normal comes from the Gaussian cache.
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pa.Array, pb.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xa = (double*)w.A, xb = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (a, b) reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double av = xa[j * w.StrideA], bv = xb[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xa, w.StrideA, xb, w.StrideB);
                            var setup = new LegacyBetaSetup(av, bv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyBeta(ref src, in setup);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="scale"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d scale and no size.</returns>
        /// <exception cref="TypeError"><paramref name="scale"/> does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="ValueError">An element is negative (<c>scale &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/><c>legacy_exponential</c> = <c>scale * -log(1 - U)</c> per position: exactly one uniform per value, so the
        ///     uniforms are bulk-filled and transformed in place. Byte-identical to NumPy.
        /// </remarks>
        public NDArray exponential(NDArray scale, Shape size = default)
        {
            using var ps = RandomParam.Float64(scale, 1.0);
            if (ps.IsScalar)
                return exponential(ps.Scalar(), size);

            RandomConstraints.CheckArray(ps.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, ps.Array);
            unsafe
            {
                FillUniforms(ret);
                // The draws are all taken; legacy_exponential's expression per position over its own uniform.
                using var w = new BroadcastWalk(ret, ps.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, s = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = s[j * w.StrideA] * -Math.Log(1.0 - o[j * w.OutStride]);
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a uniform distribution over <c>[low, high)</c> with array-valued, broadcast bounds.
        /// </summary>
        /// <param name="low">Lower boundary (inclusive). Null is NumPy's default, 0.</param>
        /// <param name="high">Upper boundary (exclusive). Null is NumPy's default, 1. <c>high &lt; low</c> is legal (the
        ///     legacy sampler has no sign check) and samples <c>(high, low]</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     broadcast shape of <c>low</c> and <c>high - low</c>.</param>
        /// <returns>The float64 draws; a 0-d array when both bounds are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A bound does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="OverflowException">An element of <c>high - low</c> is infinite or NaN — NumPy's
        ///     <c>OverflowError('Range exceeds valid bounds')</c>, raised before anything is drawn.</exception>
        /// <exception cref="ValueError">The bounds do not broadcast (NumPy's <c>np.subtract</c> text: <c>operands could not be
        ///     broadcast together with shapes …</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
        ///     <br/>mtrand computes <c>np.subtract(high, low)</c> as a whole array first (its broadcast error is the ufunc's),
        ///     rejects any non-finite range, then draws <c>low + range * next_double()</c> per position over the broadcast of
        ///     <c>low</c> and the range. The uniforms are bulk-filled (one per value) and transformed in place.
        ///     Byte-identical to NumPy.
        /// </remarks>
        public NDArray uniform(NDArray low, NDArray high = null, Shape size = default)
        {
            using var pLow = RandomParam.Float64(low, 0.0);
            using var pHigh = RandomParam.Float64(high, 1.0);
            if (pLow.IsScalar && pHigh.IsScalar)
                return uniform(pLow.Scalar(), pHigh.Scalar(), size);

            using var rangeArr = RandomBroadcast.Ufunc(() => np.subtract(pHigh.Array, pLow.Array));
            using var pRange = RandomParam.Float64(rangeArr);
            // mtrand: `if not np.all(np.isfinite(arange)): raise OverflowError(...)` — and nothing else (the Generator's
            // CONS_NON_NEGATIVE on the range does not exist here).
            if (!RandomBroadcast.AllInRange(pRange.Array, -double.MaxValue, double.MaxValue))
                throw new OverflowException("Range exceeds valid bounds");
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLow.Array, pRange.Array);
            unsafe
            {
                FillUniforms(ret);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c> (complex).</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        ///     <br/><c>legacy_normal</c> = <c>loc + scale * legacy_gauss</c> per position: one polar normal per value
        ///     whatever the parameters, so the normals come from the two-phase polar fill (which consumes and leaves
        ///     RandomState's cached Gaussian exactly as the per-value calls would) and are transformed in place.
        ///     Byte-identical to NumPy.
        /// </remarks>
        public NDArray normal(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return normal(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLoc.Array, pScale.Array);
            unsafe
            {
                FillLegacyGauss(ret);
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
        /// <param name="shape">The shape parameter; sign bit must be clear, NaN propagates; 0 returns 0 without drawing.
        ///     Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="shape"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d shape and no size.</returns>
        /// <exception cref="TypeError"><paramref name="shape"/> does not cast to float64 under <c>'safe'</c>, or a
        ///     <c>None</c> reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is negative (<c>shape &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
        ///     <br/><c>legacy_standard_gamma</c> per position (Marsaglia-Tsang over the cached-Gaussian polar normals above
        ///     1, Johnk/Ahrens-Dieter below). Byte-identical to NumPy.
        /// </remarks>
        public NDArray standard_gamma(NDArray shape, Shape size = default)
        {
            using var p = RandomParam.Float64(shape);
            if (p.IsScalar)
                return standard_gamma(p.ScalarChecked("shape", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(p.Array, "shape", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // A zero shape draws nothing; any such element forbids the read-ahead (capacity 1 = per draw).
                bool allDraw = !RandomBroadcast.AnyZero(p.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (randomizer.@lock)
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
                                o[j * w.OutStride] = LegacyStandardGamma(ref src, in g);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="shape"/> (checked first) or <paramref name="scale"/>
        ///     is negative, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
        ///     <br/><c>legacy_gamma</c> = <c>scale * legacy_standard_gamma(shape)</c> per position (a zero shape returns 0
        ///     without drawing). Byte-identical to NumPy.
        /// </remarks>
        public NDArray gamma(NDArray shape, NDArray scale = null, Shape size = default)
        {
            using var pShape = RandomParam.Float64(shape);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pShape.IsScalar && pScale.IsScalar)
                return gamma(pShape.ScalarChecked("shape", ConstraintType.CONS_NON_NEGATIVE), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pShape.Array, "shape", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pShape.Array, pScale.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = !RandomBroadcast.AnyZero(pShape.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pShape.Array, pScale.Array);
                lock (randomizer.@lock)
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
                                o[j * w.OutStride] = LegacyGamma(ref src, in g, sc[j * w.StrideB]);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="dfnum"/> (checked first) or <paramref name="dfden"/>
        ///     is not positive, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html
        ///     <br/><c>legacy_f</c> per position, the numerator chi-square first. Byte-identical to NumPy.
        /// </remarks>
        public NDArray f(NDArray dfnum, NDArray dfden, Shape size = default)
        {
            using var pNum = RandomParam.Float64(dfnum);
            using var pDen = RandomParam.Float64(dfden);
            if (pNum.IsScalar && pDen.IsScalar)
                return f(pNum.ScalarChecked("dfnum", ConstraintType.CONS_POSITIVE), pDen.ScalarChecked("dfden", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pNum.Array, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pDen.Array, "dfden", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pNum.Array, pDen.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // A position draws unless BOTH chi-squares' df / 2 are 0 (LegacyFSetup.Draws); decided conservatively from
                // the elements — when either parameter never halves to 0, every position draws.
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(pNum.Array) || !RandomBroadcast.AnyHalvesToZero(pDen.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pNum.Array, pDen.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xn = (double*)w.A, xd = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (dfnum, dfden) reuses both chi-square setups (bit-neutral, see RandomBroadcast.RunEnd).
                            double nv = xn[j * w.StrideA], dv = xd[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xd, w.StrideB);
                            var setup = new LegacyFSetup(nv, dv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyF(ref src, in setup);
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
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN accepted, drawn, and returned as NaN).
        ///     Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when every parameter is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="dfnum"/> / <paramref name="dfden"/> not positive,
        ///     <paramref name="nonc"/> negative; or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html
        ///     <br/><c>legacy_noncentral_f</c> per position. Byte-identical to NumPy.
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
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pNum.Array, pDen.Array, pNonc.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // A position draws when its denominator chi-square does (dfden / 2 != 0) or its numerator does.
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(pDen.Array) || LegacyNoncentralChisquareAllDraw(pNum.Array, pNonc.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pNum.Array, pDen.Array, pNonc.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xn = (double*)w.A, xd = (double*)w.B, xc = (double*)w.C;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (dfnum, dfden, nonc) reuses both setups (bit-neutral, see RandomBroadcast.RunEnd).
                            double nv = xn[j * w.StrideA], dv = xd[j * w.StrideB], cv = xc[j * w.StrideC];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xd, w.StrideB, xc, w.StrideC);
                            var setup = new LegacyNoncentralFSetup(nv, dv, cv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyNoncentralF(ref src, in setup);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="df"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="df"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="df"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>df &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html
        ///     <br/><c>legacy_chisquare</c> = <c>2 * legacy_standard_gamma(df / 2)</c> per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray chisquare(NDArray df, Shape size = default)
        {
            using var p = RandomParam.Float64(df);
            if (p.IsScalar)
                return chisquare(p.ScalarChecked("df", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "df", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // df > 0 (or NaN) draws unless df / 2 underflows to 0 (the gamma of shape 0 returns 0 without drawing).
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(p.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (randomizer.@lock)
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
                                o[j * w.OutStride] = LegacyChisquare(ref src, in half);
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
        /// <param name="nonc">Non-centrality, non-negative (<c>-0.0</c> rejected; NaN accepted — it draws, then returns
        ///     NaN, the legacy stream-preserving guard). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="df"/> (checked first) is not positive or of
        ///     <paramref name="nonc"/> negative, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html
        ///     <br/><c>legacy_noncentral_chisquare</c> per position. Unlike the Generator's sampler a NaN non-centrality
        ///     still draws (the NaN guard sits after the draws). Byte-identical to NumPy.
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
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pDf.Array, pNonc.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = LegacyNoncentralChisquareAllDraw(pDf.Array, pNonc.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pDf.Array, pNonc.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xd = (double*)w.A, xc = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (df, nonc) reuses its setup (bit-neutral, see RandomBroadcast.RunEnd).
                            double dv = xd[j * w.StrideA], cv = xc[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xd, w.StrideA, xc, w.StrideB);
                            var setup = new LegacyNoncentralChisquareSetup(dv, cv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyNoncentralChisquare(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Whether EVERY broadcast position of <c>legacy_noncentral_chisquare</c> is guaranteed to draw at least once,
        ///     decided from the parameter elements alone — the condition a read-ahead needs.
        /// </summary>
        /// <param name="df">The converted degrees of freedom (dense float64, <see cref="RandomParam.Array"/>).</param>
        /// <param name="nonc">The converted non-centralities (dense float64, <see cref="RandomParam.Array"/>).</param>
        /// <returns>True when no position can draw nothing; false when some position might (the caller then reads per draw).</returns>
        /// <remarks>
        ///     <para>
        ///     A legacy position draws nothing exactly when <c>df / 2 == 0</c> AND <c>nonc / 2 == 0</c>: with
        ///     <c>nonc == 0</c> it is a chi-square of a shape-0 gamma; with <c>df &lt;= 1</c> and a non-centrality that halves
        ///     to 0, the Poisson of mean 0 returns 0 without drawing and the chi-square of <c>df + 0</c> does too. The
        ///     <c>df &gt; 1</c> branch always draws (its gamma's uniform), and a NaN non-centrality draws its Poisson uniform
        ///     (NaN never halves to 0) — unlike the Generator's sampler, which returns NaN first.
        ///     </para>
        ///     <para>
        ///     Conservative over the elements (a failing pair needs both kinds of element), which is sound: false only
        ///     costs the read-ahead.
        ///     </para>
        /// </remarks>
        private static bool LegacyNoncentralChisquareAllDraw(NDArray df, NDArray nonc)
            => !RandomBroadcast.AnyHalvesToZero(df) || !RandomBroadcast.AnyHalvesToZero(nonc);

        /// <summary>
        ///     Draw samples from a standard Student's t distribution with array-valued, broadcast degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="df"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="df"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="df"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>df &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
        ///     <br/><c>legacy_standard_t</c> = <c>sqrt(df/2) * N / sqrt(G(df/2))</c> per position, the polar normal drawn
        ///     first. Byte-identical to NumPy.
        /// </remarks>
        public NDArray standard_t(NDArray df, Shape size = default)
        {
            using var p = RandomParam.Float64(df);
            if (p.IsScalar)
                return standard_t(p.ScalarChecked("df", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "df", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // The normal may come from the Gaussian cache (no draw), so only the gamma guarantees one: a df halving
                // to 0 makes that gamma draw-free and forbids the read-ahead.
                bool allDraw = !RandomBroadcast.AnyHalvesToZero(p.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, d = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal df reuses its gamma setup and sqrt(df / 2) (bit-neutral, see RandomBroadcast.RunEnd).
                            double dv = d[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, d, w.StrideA);
                            var setup = new LegacyStandardTSetup(dv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyStandardT(ref src, in setup);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws in <c>[-pi, pi]</c>; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="kappa"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html
        ///     <br/><c>legacy_vonmises</c> per position. A kappa large enough that <c>4 * kappa^2</c> overflows (infinity
        ///     included) never returns in NumPy (its rejection test compares NaN); like the scalar sampler this returns
        ///     <c>mu</c> wrapped to <c>[-pi, pi]</c> without drawing. Otherwise byte-identical to NumPy.
        /// </remarks>
        public NDArray vonmises(NDArray mu, NDArray kappa, Shape size = default)
        {
            using var pMu = RandomParam.Float64(mu);
            using var pKappa = RandomParam.Float64(kappa);
            if (pMu.IsScalar && pKappa.IsScalar)
                return vonmises(pMu.Scalar(), pKappa.ScalarChecked("kappa", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(pKappa.Array, "kappa", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pMu.Array, pKappa.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // Whether a position draws depends on kappa alone: NaN, or a kappa whose envelope overflows (from
                // LegacyVonmisesOverflowKappa up), draws nothing (LegacyVonmisesSetup.Draws).
                bool allDraw = !RandomBroadcast.AnyNaN(pKappa.Array)
                               && !RandomBroadcast.AnyInRange(pKappa.Array, LegacyVonmisesOverflowKappa, double.PositiveInfinity);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pMu.Array, pKappa.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, xm = (double*)w.A, xk = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (mu, kappa) reuses its envelope (bit-neutral, see RandomBroadcast.RunEnd).
                            double mv = xm[j * w.StrideA], kv = xk[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xm, w.StrideA, xk, w.StrideB);
                            var setup = new LegacyVonmisesSetup(mv, kv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyVonmises(ref src, in setup);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>a &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html
        ///     <br/><c>legacy_pareto</c> = <c>exp(-log(1 - U) / a) - 1</c> per position over bulk-filled uniforms (the legacy
        ///     <c>exp(x) - 1</c>, not <c>expm1</c> — so, unlike the Generator's, byte-identical to NumPy).
        /// </remarks>
        public NDArray pareto(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return pareto(p.ScalarChecked("a", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                FillUniforms(ret);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = Math.Exp(-Math.Log(1.0 - o[j * w.OutStride]) / x[j * w.StrideA]) - 1;
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Weibull distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="a">Shape, non-negative (<c>-0.0</c> rejected; NaN accepted); 0 returns 0 without drawing. Null is
        ///     Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is negative (<c>a &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
        ///     <br/><c>legacy_weibull</c> = <c>(-log(1 - U))^(1/a)</c> per position, 0 without a draw where <c>a == 0</c>.
        ///     The uniforms the drawing positions take are bulk-filled into the TAIL of the output and consumed front to
        ///     back — a position never reads a slot an earlier position overwrote, because at most as many positions precede
        ///     it as draw-free positions exist. Byte-identical to NumPy.
        /// </remarks>
        public NDArray weibull(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return weibull(p.ScalarChecked("a", ConstraintType.CONS_NON_NEGATIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
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
                if (draws > 0)
                    lock (randomizer.@lock)
                        randomizer.FillDouble(dst + (n - draws), draws);
                long next = n - draws;
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                    {
                        double av = x[j * w.StrideA];
                        // legacy_weibull: a == 0 returns 0 before drawing; else pow(legacy_standard_exponential, 1. / a).
                        o[j * w.OutStride] = av == 0.0 ? 0.0 : Math.Pow(-Math.Log(1.0 - dst[next++]), 1.0 / av);
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples in <c>[0, 1]</c> from a power distribution with an array-valued, broadcast exponent.
        /// </summary>
        /// <param name="a">Parameter, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="a"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not positive (<c>a &lt;= 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html
        ///     <br/><c>legacy_power</c> = <c>(1 - exp(-E))^(1/a)</c> with <c>E = -log(1 - U)</c> per position over bulk-filled
        ///     uniforms (the legacy <c>1 - exp</c>, not <c>-expm1</c> — byte-identical to NumPy).
        /// </remarks>
        public NDArray power(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return power(p.ScalarChecked("a", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                FillUniforms(ret);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, x = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                    {
                        double e = -Math.Log(1.0 - o[j * w.OutStride]);
                        o[j * w.OutStride] = Math.Pow(1 - Math.Exp(-e), 1.0 / x[j * w.StrideA]);
                    }
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html
        ///     <br/>mtrand calls the modern <c>random_laplace</c> on the bare bit generator per position (one uniform,
        ///     redrawn only on an exact 0). Byte-identical to NumPy.
        /// </remarks>
        public NDArray laplace(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return laplace(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLoc.Array, pScale.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // every value draws
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (randomizer.@lock)
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html
        ///     <br/>mtrand calls the modern <c>random_gumbel</c> on the bare bit generator per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray gumbel(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return gumbel(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLoc.Array, pScale.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // every value draws
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (randomizer.@lock)
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="scale"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html
        ///     <br/>mtrand calls the modern <c>random_logistic</c> on the bare bit generator per position. Byte-identical to NumPy.
        /// </remarks>
        public NDArray logistic(NDArray loc, NDArray scale = null, Shape size = default)
        {
            using var pLoc = RandomParam.Float64(loc, 0.0);
            using var pScale = RandomParam.Float64(scale, 1.0);
            if (pLoc.IsScalar && pScale.IsScalar)
                return logistic(pLoc.Scalar(), pScale.Scalar(), size);

            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLoc.Array, pScale.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // every value draws
                using var w = new BroadcastWalk(ret, pLoc.Array, pScale.Array);
                lock (randomizer.@lock)
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element of <paramref name="sigma"/> is negative, or the shapes do not broadcast /
        ///     the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
        ///     <br/><c>legacy_lognormal</c> = <c>exp(mean + sigma * legacy_gauss)</c> per position over the two-phase polar
        ///     fill (Gaussian cache consumed and left as the per-value calls would). Byte-identical to NumPy.
        /// </remarks>
        public NDArray lognormal(NDArray mean, NDArray sigma = null, Shape size = default)
        {
            using var pMean = RandomParam.Float64(mean, 0.0);
            using var pSigma = RandomParam.Float64(sigma, 1.0);
            if (pMean.IsScalar && pSigma.IsScalar)
                return lognormal(pMean.Scalar(), pSigma.Scalar(), size);

            RandomConstraints.CheckArray(pSigma.Array, "sigma", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pMean.Array, pSigma.Array);
            unsafe
            {
                FillLegacyGauss(ret);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="scale"/>'s shape.</param>
        /// <returns>The float64 draws; a 0-d array for a 0-d scale and no size.</returns>
        /// <exception cref="TypeError"><paramref name="scale"/> does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element is negative (<c>scale &lt; 0</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html
        ///     <br/><c>legacy_rayleigh</c> = <c>mode * sqrt(-2 * log1p(-U))</c> per position over bulk-filled uniforms (it
        ///     takes the bare bit generator, so the Gaussian cache is untouched). Byte-identical to NumPy.
        /// </remarks>
        public NDArray rayleigh(NDArray scale, Shape size = default)
        {
            using var p = RandomParam.Float64(scale, 1.0);
            if (p.IsScalar)
                return rayleigh(p.Scalar(), size);

            RandomConstraints.CheckArray(p.Array, "scale", ConstraintType.CONS_NON_NEGATIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, p.Array);
            unsafe
            {
                FillUniforms(ret);
                using var w = new BroadcastWalk(ret, p.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, s = (double*)w.A;
                    for (long j = 0; j < w.Count; j++)
                        o[j * w.OutStride] = s[j * w.StrideA] * Math.Sqrt(-2.0 * global::NumSharp.Generator.Log1p(-o[j * w.OutStride]));
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Wald (inverse Gaussian) distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="mean">Distribution mean, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="scale">Scale, positive (NaN accepted). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when both parameters are 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches
        ///     the scalar path.</exception>
        /// <exception cref="ValueError">An element of <paramref name="mean"/> (checked first) or <paramref name="scale"/> is
        ///     not positive, or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html
        ///     <br/><c>legacy_wald</c> per position (a cached-Gaussian polar normal and one acceptance uniform), with the
        ///     legacy cancellation-prone spelling the stream depends on. Byte-identical to NumPy.
        /// </remarks>
        public NDArray wald(NDArray mean, NDArray scale, Shape size = default)
        {
            using var pMean = RandomParam.Float64(mean);
            using var pScale = RandomParam.Float64(scale);
            if (pMean.IsScalar && pScale.IsScalar)
                return wald(pMean.ScalarChecked("mean", ConstraintType.CONS_POSITIVE), pScale.ScalarChecked("scale", ConstraintType.CONS_POSITIVE), size);

            RandomConstraints.CheckArray(pMean.Array, "mean", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pScale.Array, "scale", ConstraintType.CONS_POSITIVE);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pMean.Array, pScale.Array);
            unsafe
            {
                long n = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // Every value draws its acceptance uniform; a refill inside a polar attempt still owes that attempt's
                // second draw on top of every later uniform, so "one per value still owed" stays a lower bound.
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pMean.Array, pScale.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        double* o = (double*)w.Out, m = (double*)w.A, s = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (mean, scale) reuses mu_2l (bit-neutral, see RandomBroadcast.RunEnd).
                            double mv = m[j * w.StrideA], sv = s[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, m, w.StrideA, s, w.StrideB);
                            var setup = new LegacyWaldSetup(mv, sv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = n - i;
                                o[j * w.OutStride] = LegacyWald(ref src, in setup);
                            }
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     limits' broadcast shape.</param>
        /// <returns>The float64 draws; a 0-d array when every limit is 0-d and no size is given.</returns>
        /// <exception cref="TypeError">A limit does not cast to float64 under <c>'safe'</c>, or a <c>None</c> reaches the
        ///     scalar path (read in NumPy's order: left, right, mode).</exception>
        /// <exception cref="ValueError">In NumPy's order: any <c>left &gt; mode</c>, any <c>mode &gt; right</c>, any
        ///     <c>left == right</c>; the comparisons' own broadcast errors (<c>operands could not be broadcast together with
        ///     shapes …</c>); or an incompatible size.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
        ///     <br/>mtrand's three whole-array comparisons (<c>np.greater</c> / <c>np.equal</c>), then
        ///     <c>cont_broadcast_3</c> over <c>random_triangular</c>: exactly one uniform per value, so the uniforms are
        ///     bulk-filled and each position applies <c>random_triangular</c>'s per-call statements to its own. A NaN limit
        ///     passes every comparison and samples NaN. Byte-identical to NumPy.
        /// </remarks>
        public NDArray triangular(NDArray left, NDArray mode, NDArray right, Shape size = default)
        {
            using var pLeft = RandomParam.Float64(left);
            using var pMode = RandomParam.Float64(mode);
            using var pRight = RandomParam.Float64(right);
            if (pLeft.IsScalar && pMode.IsScalar && pRight.IsScalar)
            {
                // mtrand reads fleft, fright, fmode — in that order — before any comparison.
                double fleft = pLeft.Scalar(), fright = pRight.Scalar(), fmode = pMode.Scalar();
                return triangular(fleft, fmode, fright, size);
            }

            if (RandomBroadcast.AnyCompare(NDExpr.Greater, pLeft.Array, pMode.Array))
                throw new ValueError("left > mode");
            if (RandomBroadcast.AnyCompare(NDExpr.Greater, pMode.Array, pRight.Array))
                throw new ValueError("mode > right");
            if (RandomBroadcast.AnyCompare(NDExpr.Equal, pLeft.Array, pRight.Array))
                throw new ValueError("left == right");
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Double, size, !size.IsEmpty, pLeft.Array, pMode.Array, pRight.Array);
            unsafe
            {
                FillUniforms(ret);
                using var w = new BroadcastWalk(ret, pLeft.Array, pMode.Array, pRight.Array);
                while (w.Next())
                {
                    double* o = (double*)w.Out, l = (double*)w.A, m = (double*)w.B, r = (double*)w.C;
                    for (long j = 0; j < w.Count; j++)
                    {
                        // random_triangular's per-call statements (Distributions.RandomTriangular), over this position's uniform.
                        double lv = l[j * w.StrideA], mv = m[j * w.StrideB], rv = r[j * w.StrideC];
                        double @base = rv - lv;
                        double leftbase = mv - lv;
                        double ratio = leftbase / @base;
                        double leftprod = leftbase * @base;
                        double rightprod = (rv - mv) * @base;
                        double U = o[j * w.OutStride];
                        o[j * w.OutStride] = U <= ratio
                            ? lv + Math.Sqrt(U * leftprod)
                            : rv - Math.Sqrt((1.0 - U) * rightprod);
                    }
                }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a binomial distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="n">Number of trials, a non-negative INTEGER array (NumPy's <c>intp</c>; a float array is NumPy's
        ///     safe-cast <see cref="TypeError"/>). Null is Python's <c>None</c> (a <see cref="TypeError"/>).</param>
        /// <param name="p">Probability of success, in <c>[0, 1]</c> (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The int64 draws (NumPy's C long, modelled LP64); a 0-d array when both parameters are 0-d and no size
        ///     is given.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> (converted first) does not cast to float64 or
        ///     <paramref name="n"/> to int64 under <c>'safe'</c>, or a <c>None</c> reaches a conversion / the scalar path.</exception>
        /// <exception cref="ValueError">In NumPy's order: an element of <paramref name="p"/> outside <c>[0, 1]</c> or NaN,
        ///     then of <paramref name="n"/> negative; or the shapes do not broadcast / the size is incompatible (NumPy numbers
        ///     <paramref name="p"/> before <paramref name="n"/> in the mismatch text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        ///     <br/>mtrand's own broadcast loop over <c>legacy_random_binomial</c> with RandomState's binomial setup cache
        ///     (<c>self._binomial</c>, repopulated whenever consecutive positions change <c>(n, p)</c>, exactly as NumPy's).
        ///     The legacy sampler has no <c>n == 0</c> / <c>p == 0</c> shortcut, so every position draws. Byte-identical to
        ///     NumPy's LP64 build (Windows NumPy's 32-bit long bounds <c>n</c> at <c>2^31 - 1</c>).
        /// </remarks>
        public NDArray binomial(NDArray n, NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            using var pn = RandomParam.Int64(n);
            if (pp.IsScalar && pn.IsScalar)
                return binomial(pn.ScalarInt64(), pp.Scalar(), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_0_1);
            RandomConstraints.CheckArray(pn.Array, "n", ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, pp.Array, pn.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // inversion or BTPE: >= 1 draw
                using var w = new BroadcastWalk(ret, pp.Array, pn.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out, xn = (long*)w.B;
                        double* xp = (double*)w.A;
                        for (long j = 0; j < w.Count; j++, i++)
                        {
                            src.Owed = count - i;
                            o[j * w.OutStride] = LegacyBinomial(ref src, xp[j * w.StrideA], xn[j * w.StrideB]);
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a negative binomial distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="n">Parameter, positive (NaN accepted by the legacy constraint; need not be an integer). Null is
        ///     Python's <c>None</c>.</param>
        /// <param name="p">Probability of success, in <c>[0, 1]</c> (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The int64 draws (NumPy's C long, modelled LP64); a 0-d array when both parameters are 0-d and no size
        ///     is given.</returns>
        /// <exception cref="TypeError"><paramref name="n"/> (converted first) or <paramref name="p"/> does not cast to
        ///     float64 under <c>'safe'</c>, or a <c>None</c> reaches the scalar path.</exception>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="n"/> not positive, <paramref name="p"/> outside
        ///     <c>[0, 1]</c> or NaN; or the shapes do not broadcast / the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html
        ///     <br/><c>disc(&amp;legacy_negative_binomial, …)</c>: a Poisson of a legacy gamma with scale <c>(1 - p) / p</c> per
        ///     position. Unlike the Generator there is no Poisson-mean bound; <c>p == 0</c> makes the mean infinite and the
        ///     count C's integer cast of it. Byte-identical to NumPy's LP64 build.
        /// </remarks>
        public NDArray negative_binomial(NDArray n, NDArray p, Shape size = default)
        {
            using var pn = RandomParam.Float64(n);
            using var pp = RandomParam.Float64(p);
            if (pn.IsScalar && pp.IsScalar)
                return negative_binomial(pn.ScalarChecked("n", ConstraintType.CONS_POSITIVE), pp.ScalarChecked("p", ConstraintType.CONS_BOUNDED_0_1), size);

            RandomConstraints.CheckArray(pn.Array, "n", ConstraintType.CONS_POSITIVE);
            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_0_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, pn.Array, pp.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // The gamma of shape n > 0 (or NaN) always draws its uniform, even with a cached normal.
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                using var w = new BroadcastWalk(ret, pn.Array, pp.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* xn = (double*)w.A, xp = (double*)w.B;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (n, p) reuses the gamma setup and its scale (bit-neutral, see RandomBroadcast.RunEnd).
                            double nv = xn[j * w.StrideA], pv = xp[j * w.StrideB];
                            long end = RandomBroadcast.RunEnd(j, w.Count, xn, w.StrideA, xp, w.StrideB);
                            var setup = new LegacyNegativeBinomialSetup(nv, pv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = LegacyNegativeBinomial(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Poisson distribution with an array-valued, broadcast mean.
        /// </summary>
        /// <param name="lam">Expected number of events, non-negative, not NaN, at most the legacy bound
        ///     <c>long max - 10*sqrt(long max)</c>. Null is NumPy's default, 1.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="lam"/>'s shape.</param>
        /// <returns>The int64 draws (NumPy's C long, modelled LP64); a 0-d array for a 0-d mean and no size.</returns>
        /// <exception cref="TypeError"><paramref name="lam"/> does not cast to float64 under <c>'safe'</c>.</exception>
        /// <exception cref="ValueError">An element is too large or NaN (<c>lam value too large</c> — the array test runs the
        ///     bound FIRST, so NaN reports it) or negative (<c>lam &lt; 0 or lam contains NaNs</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
        ///     <br/><c>legacy_random_poisson</c> (the modern <c>random_poisson</c> on the bare bit generator) per position:
        ///     the product method below 10, PTRS above, 0 without a draw for <c>lam == 0</c>. Byte-identical to NumPy's LP64
        ///     build (Windows NumPy bounds <c>lam</c> by its 32-bit long).
        /// </remarks>
        public NDArray poisson(NDArray lam, Shape size = default)
        {
            using var p = RandomParam.Float64(lam, 1.0);
            if (p.IsScalar)
                return poisson(p.Scalar(), size);

            RandomConstraints.CheckArray(p.Array, "lam", ConstraintType.LEGACY_CONS_POISSON);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, p.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                bool allDraw = !RandomBroadcast.AnyZero(p.Array);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (randomizer.@lock)
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="a"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>; NumPy's C long, modelled LP64); a 0-d array for a 0-d
        ///     <paramref name="a"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is not above 1 or is NaN (<c>a &lt;= 1 or a contains NaNs</c>), or the
        ///     size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html
        ///     <br/><c>legacy_random_zipf</c> per position. For <c>a &gt;= 1025</c> (infinity included) NumPy's legacy loop
        ///     never returns; like the scalar sampler this returns 1 there without drawing. Otherwise byte-identical to NumPy.
        /// </remarks>
        public NDArray zipf(NDArray a, Shape size = default)
        {
            using var p = RandomParam.Float64(a);
            if (p.IsScalar)
                return zipf(p.ScalarChecked("a", ConstraintType.CONS_GT_1), size);

            RandomConstraints.CheckArray(p.Array, "a", ConstraintType.CONS_GT_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, p.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // 2^(a-1) overflowing is the draw-free input (LegacyZipfSetup.Draws), which is exactly a >= 1025: the true
                // 2^x for the largest double x below 1024 (1024 - 2^-42) sits ~1.6e-13 relative below 2^1024 — thousands of
                // ulps under DBL_MAX, so pow returns it finite — while 2^1024 itself overflows.
                bool allDraw = !RandomBroadcast.AnyInRange(p.Array, 1025.0, double.PositiveInfinity);
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, p.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal exponents reuses its setup and — when as long as the scalar fill's
                            // threshold — the acceptance test's pow(1 + 1/X, a - 1) memo, which is keyed by candidate
                            // for ONE exponent (both bit-neutral, see RandomBroadcast.RunEnd and LegacyZipf).
                            double av = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            var setup = new LegacyZipfSetup(av);
                            double[] tMemo = null;
                            if (end - j >= ZipfMemoMinFill && setup.Draws)
                            {
                                tMemo = new double[256];
                                System.Array.Fill(tMemo, double.NaN);
                            }
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = LegacyZipf(ref src, in setup, tMemo);
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="p"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>; NumPy's C long, modelled LP64); a 0-d array for a 0-d
        ///     <paramref name="p"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is outside <c>(0, 1]</c> or NaN, or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
        ///     <br/><c>legacy_random_geometric</c> per position: the CDF search for <c>p &gt;= 1/3</c>, else the legacy
        ///     inversion <c>ceil(log1p(-U) / log(1 - p))</c>; one uniform either way. Byte-identical to NumPy's LP64 build.
        /// </remarks>
        public NDArray geometric(NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            if (pp.IsScalar)
                return geometric(pp.ScalarChecked("p", ConstraintType.CONS_BOUNDED_GT_0_1), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_GT_0_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, pp.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // one uniform per value
                using var w = new BroadcastWalk(ret, pp.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal p reuses log(1 - p) (bit-neutral, see RandomBroadcast.RunEnd).
                            double pv = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            var setup = new LegacyGeometricSetup(pv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = LegacyGeometric(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution with array-valued, broadcast parameters.
        /// </summary>
        /// <param name="ngood">Good selections, a non-negative INTEGER array. Null is Python's <c>None</c> (a
        ///     <see cref="TypeError"/>).</param>
        /// <param name="nbad">Bad selections, a non-negative integer array.</param>
        /// <param name="nsample">Items sampled, <c>1 &lt;= nsample &lt;= ngood + nbad</c> per position.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is the
        ///     parameters' broadcast shape.</param>
        /// <returns>The int64 draws (NumPy's C long, modelled LP64); a 0-d array when every parameter is 0-d and no size
        ///     is given.</returns>
        /// <exception cref="TypeError">A parameter does not cast to int64 under <c>'safe'</c> (floats, uint64, complex) or is
        ///     <c>None</c>.</exception>
        /// <exception cref="ValueError">In NumPy's order: any <c>ngood + nbad &lt; nsample</c> (whose whole-array arithmetic
        ///     raises the ufunc broadcast error for mismatched shapes); then a negative <paramref name="ngood"/>, a negative
        ///     <paramref name="nbad"/>, an <paramref name="nsample"/> below 1; or an incompatible size.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html
        ///     <br/><c>discrete_broadcast_iii</c> over <c>legacy_random_hypergeometric</c>: HRUA for <c>nsample &gt; 10</c>,
        ///     the legacy urn simulation for <c>1..10</c>, each with its setup per call as NumPy's. Unlike the Generator there
        ///     are no <c>10^9</c> caps. Byte-identical to NumPy's LP64 build (Windows NumPy bounds <c>ngood</c> by its 32-bit
        ///     long).
        /// </remarks>
        public NDArray hypergeometric(NDArray ngood, NDArray nbad, NDArray nsample, Shape size = default)
        {
            using var pg = RandomParam.Int64(ngood);
            using var pb = RandomParam.Int64(nbad);
            using var ps = RandomParam.Int64(nsample);
            if (pg.IsScalar && pb.IsScalar && ps.IsScalar)
                return hypergeometric(pg.ScalarInt64(), pb.ScalarInt64(), ps.ScalarInt64(), size);

            // mtrand: `if np.any(np.less(np.add(ongood, onbad), onsample))` BEFORE the per-parameter constraints; the int64
            // add wraps as NumPy's does.
            using (var total = RandomBroadcast.Ufunc(() => np.add(pg.Array, pb.Array)))
                if (RandomBroadcast.AnyAndDispose(RandomBroadcast.Ufunc(() => np.less(total, ps.Array))))
                    throw new ValueError("ngood + nbad < nsample");
            RandomConstraints.CheckArray(pg.Array, "ngood", ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG);
            RandomConstraints.CheckArray(pb.Array, "nbad", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.CheckArray(ps.Array, "nsample", ConstraintType.CONS_GTE_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, pg.Array, pb.Array, ps.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                // HRUA (nsample > 10) always draws; the urn walk draws unless a colour is absent (its answer is then
                // known). Conservatively from the elements: every sample above 10, or no zero good/bad count anywhere.
                bool allDraw = !RandomBroadcast.AnyInRange(ps.Array, double.NegativeInfinity, 10.0)
                               || (!RandomBroadcast.AnyZero(pg.Array) && !RandomBroadcast.AnyZero(pb.Array));
                var src = new DrawBufferDouble(randomizer, storage, allDraw ? DrawBufferDouble.Capacity : 1);
                using var w = new BroadcastWalk(ret, pg.Array, pb.Array, ps.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out, g = (long*)w.A, b = (long*)w.B, s = (long*)w.C;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal (ngood, nbad, nsample) takes one branch of NumPy's dispatch and shares what the
                            // scalar fill of the same length shares: HRUA's setup and — past the memo threshold — its
                            // loggam-sum memo, or the urn walk's ratio table (all bit-neutral: RandomBroadcast.RunEnd,
                            // LegacyHruaSetup, LoggamSumMemo, LegacyHypTable).
                            long gv = g[j * w.StrideA], bv = b[j * w.StrideB], sv = s[j * w.StrideC];
                            long end = RandomBroadcast.RunEnd(j, w.Count, g, w.StrideA, b, w.StrideB, s, w.StrideC);
                            bool repays = end - j >= HypergeometricMemoMinFill;
                            if (sv > 10)
                            {
                                var hruaSetup = new LegacyHruaSetup(gv, bv, sv);
                                var memo = repays ? new LoggamSumMemo() : null;
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = LegacyHypergeometricHrua(ref src, gv, bv, sv, in hruaSetup, memo);
                                }
                            }
                            else
                            {
                                // nsample is validated >= 1 (CONS_GTE_1), so the urn walk is NumPy's branch here.
                                var table = repays && LegacyHypTable.Applicable(gv, bv) ? new LegacyHypTable(gv, bv, sv) : null;
                                for (; j < end; j++, i++)
                                {
                                    src.Owed = count - i;
                                    o[j * w.OutStride] = table != null
                                        ? LegacyHypergeometricHyp(ref src, gv, bv, sv, table)
                                        : LegacyHypergeometric(ref src, gv, bv, sv);
                                }
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Draw samples from a logarithmic series distribution with an array-valued, broadcast shape.
        /// </summary>
        /// <param name="p">Shape, in <c>[0, 1)</c> (NaN rejected). Null is Python's <c>None</c>.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) is
        ///     <paramref name="p"/>'s shape.</param>
        /// <returns>The int64 draws (<c>&gt;= 1</c>; NumPy's C long, modelled LP64); a 0-d array for a 0-d
        ///     <paramref name="p"/> and no size.</returns>
        /// <exception cref="TypeError"><paramref name="p"/> does not cast to float64 under <c>'safe'</c>, or a <c>None</c>
        ///     reaches the scalar path.</exception>
        /// <exception cref="ValueError">An element is outside <c>[0, 1)</c> or NaN (<c>p &lt; 0, p &gt;= 1 or p contains
        ///     NaNs</c>), or the size is incompatible.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html
        ///     <br/><c>legacy_logseries</c> per position (the legacy <c>log(1 - p)</c> / <c>1 - exp(r*U)</c> spellings, so —
        ///     unlike the Generator's — byte-identical to NumPy).
        /// </remarks>
        public NDArray logseries(NDArray p, Shape size = default)
        {
            using var pp = RandomParam.Float64(p);
            if (pp.IsScalar)
                return logseries(pp.ScalarChecked("p", ConstraintType.CONS_BOUNDED_LT_0_1), size);

            RandomConstraints.CheckArray(pp.Array, "p", ConstraintType.CONS_BOUNDED_LT_0_1);
            var ret = RandomBroadcast.NewOutput(NPTypeCode.Int64, size, !size.IsEmpty, pp.Array);
            unsafe
            {
                long count = ret.size, i = 0;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity); // V is always drawn
                using var w = new BroadcastWalk(ret, pp.Array);
                lock (randomizer.@lock)
                    while (w.Next())
                    {
                        long* o = (long*)w.Out;
                        double* x = (double*)w.A;
                        for (long j = 0; j < w.Count;)
                        {
                            // A run of equal p reuses r = log(1 - p) (bit-neutral, see RandomBroadcast.RunEnd).
                            double pv = x[j * w.StrideA];
                            long end = RandomBroadcast.RunEnd(j, w.Count, x, w.StrideA);
                            var setup = new LegacyLogseriesSetup(pv);
                            for (; j < end; j++, i++)
                            {
                                src.Owed = count - i;
                                o[j * w.OutStride] = LegacyLogseries(ref src, in setup);
                            }
                        }
                    }
            }
            return ret;
        }

        /// <summary>
        ///     Bulk-fills a fresh broadcast output with one <c>next_double</c> per element, in C order under the bit
        ///     generator's lock — the draws of the samplers that take exactly one uniform per value, which then transform
        ///     each element in place.
        /// </summary>
        /// <param name="ret">The fresh, C-contiguous output (from <see cref="RandomBroadcast.NewOutput"/>).</param>
        /// <remarks>Element <c>i</c> receives NumPy's <c>i</c>-th draw because the output is C-contiguous and the walk
        ///     visits its positions in C order.</remarks>
        private unsafe void FillUniforms(NDArray ret)
        {
            long n = ret.size;
            if (n == 0)
                return;
            lock (randomizer.@lock)
                randomizer.FillDouble((double*)ret.Address, n);
        }

        /// <summary>
        ///     Bulk-fills a fresh broadcast output with one <c>legacy_gauss</c> per element (the two-phase polar fill), in C
        ///     order under the bit generator's lock — consuming and leaving RandomState's cached Gaussian exactly as that many
        ///     consecutive <c>legacy_gauss</c> calls do.
        /// </summary>
        /// <param name="ret">The fresh, C-contiguous output (from <see cref="RandomBroadcast.NewOutput"/>).</param>
        private unsafe void FillLegacyGauss(NDArray ret)
        {
            long n = ret.size;
            if (n == 0)
                return;
            double* storage = stackalloc double[DrawBufferDouble.Capacity];
            var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
            lock (randomizer.@lock)
                LegacyGaussFill(ref src, (double*)ret.Address, n);
        }
    }
}
