using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;
// The moved statements call the public np.* functions they called inside np (np.diff, np.empty, np.subtract, np.divide,
// np.evaluate, np.all) and np's internal SliceAlongAxis, and build np's GradientResult; importing np's statics keeps them
// verbatim.
using static NumSharp.np;

namespace NumSharp
{
    // ============================== np.gradient ==============================
    // Return the gradient of an N-dimensional array, computed with second-order
    // accurate central differences in the interior and first/second-order
    // one-sided differences at the boundaries. The result has the SAME shape as
    // the input (unlike np.diff, which shrinks the axis).
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::gradient
    //
    // This is a faithful port of NumPy's own implementation, which is a pure
    // composition of slicing + arithmetic — no per-element loop of our own; the
    // loops all live in the NumSharp arithmetic operators (NDIter / IL kernels).
    // Each output axis is a fresh `otype` array whose interior and two edges are
    // filled by slice-assignment of the difference expressions.
    //
    // Two dtype/precision rules are ported EXACTLY so the result is bit-identical
    // to NumPy:
    //   * Output dtype (`otype`): a float-family input keeps its dtype
    //     (Half/Single/Double/Complex; NumSharp-only Decimal too); an
    //     integer/char input is promoted to float64 (NumPy converts f to float64
    //     first, matching np.diff-free arithmetic); a BOOLEAN input raises the
    //     same "numpy boolean subtract" TypeError NumPy raises (bool is not an
    //     integer subtype, so NumPy leaves it bool and the internal subtract
    //     fails).
    //   * Spacing precision: UNIFORM spacing feeds the stencil WEAK C# double
    //     coefficients (NEP50 weak scalars, so a float32 input stays float32),
    //     while NON-UNIFORM spacing (a coordinate array) feeds STRONG float64
    //     coefficients (NumPy converts the coordinates to float64), so the
    //     stencil runs in float64 and is cast down to `otype` on store — exactly
    //     as NumPy pre-allocates `out` and assigns the computed slices into it.
    //     A reduced-precision (float16/float32) input with NON-UNIFORM spacing
    //     additionally widens the one-sided EDGE slices to float64 (see `edgeWiden`
    //     below), because NumSharp demotes a 0-d float64 Scalar coefficient to weak
    //     against a reduced-precision operand and would otherwise compute the edge
    //     in reduced precision; the interior already runs in float64 via its 1-D
    //     float64 coefficient arrays.
    //
    // ACCEPTED 1-ULP divergence, complex128 ONLY, with a spacing whose divisor is not
    // a power of two (e.g. a per-axis scalar dx=1.5 → /3.0, or any non-uniform edge
    // /dx): gradient divides the complex difference by a real spacing, and .NET's
    // System.Numerics.Complex division (Smith's algorithm, which reduces a real
    // divisor to the component-wise a/c) rounds the last bit differently from NumPy's
    // complex divide loop (which computes (a·c+b·d)/(c²+d²) = 3a/9 for c=3). This is
    // the pre-existing complex128-binary-op limitation (NumSharp's complex byte-
    // exactness is contractual for UNARY ufuncs only), reproducible in isolation as
    // `np.divide(complexArray, 3.0)`; gradient/trapezoid merely inherit it, and
    // trapezoid avoids it because it only ever divides by the exact /2.0. float16/
    // float32/float64 gradient is bit-exact.

    /// <summary>
    ///     The body of NumPy's <c>gradient</c> (<c>numpy/lib/_function_base_impl.py</c>): the axes, the spacing per axis
    ///     (scalars, or coordinate arrays reduced to their diffs), the output dtype, and per axis the interior's central
    ///     differences and the edges' one-sided differences, written into a fresh array. The public <c>np.gradient</c>
    ///     overloads only spread their spacings into one list, apply the matrix boundary and call <see cref="Gradient"/>.
    /// </summary>
    /// <remarks>
    ///     Moved from <c>Math/np.gradient.cs</c> statement for statement by the np-layer dtype plan's P2 (an operation
    ///     class, decision D1: NumPy writes gradient in Python). It sits beside its facade rather than under
    ///     <c>Backends/</c>, as <c>Math/TrapezoidEngine</c> does: its statements call np.* functions, which ratchet R2
    ///     counts against engine files.
    /// </remarks>
    internal static class GradientEngine
    {
        // ---- Core algorithm (port of numpy.gradient) ----

        /// <summary>
        ///     NumPy's <c>gradient(f, *varargs, axis=axis, edge_order=edge_order)</c>: one fresh array per requested axis,
        ///     second-order central differences inside and first- or second-order one-sided differences at both ends.
        /// </summary>
        /// <param name="f">The sampled values (the facade has applied the matrix boundary; null is refused here).</param>
        /// <param name="varargs">NumPy's <c>*varargs</c>: no entry (unit steps), one scalar for every axis, or one entry
        ///     per requested axis, each a scalar step or a 1-D coordinate array as long as its axis.</param>
        /// <param name="axisArg">The axes to differentiate along (negative values count from the end), or null for every
        ///     axis of <paramref name="f"/>.</param>
        /// <param name="edge_order">The boundary order: 1 gives first-order ends; any other value up to 2 (0 and negative
        ///     values included, as NumPy reads them) gives second-order ends.</param>
        /// <returns>One component per requested axis, a single array when exactly one axis was requested (NumPy's bare
        ///     ndarray), none for a 0-d <paramref name="f"/> without an axis.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="f"/> is null, or a spacing entry is (NumSharp's refusal
        ///     where NumPy computes <c>2. * None</c> and raises TypeError).</exception>
        /// <exception cref="AxisError">An axis outside <c>[-f.ndim, f.ndim)</c>.</exception>
        /// <exception cref="ValueError">A repeated axis; a ragged list spacing (np.ndim's inhomogeneous-shape text); a
        ///     coordinate array of another rank than 1 or another length than its axis; <paramref name="edge_order"/> above
        ///     2; an axis shorter than <c>edge_order + 1</c>.</exception>
        /// <exception cref="NotSupportedException">A list spacing holding None or another object, or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have).</exception>
        /// <exception cref="TypeError">A spacing count other than 0, 1 or the number of axes ("invalid number of
        ///     arguments"); a bool <paramref name="f"/> (NumPy's boolean subtract); a text <paramref name="f"/> (NumPy's
        ///     no-loop texts of the subtract or the coefficient multiply); bool coordinates (the boolean negative of the
        ///     coefficients); a spacing <see cref="Convert.ToDouble(object)"/> refuses (a <see cref="Half"/> or a
        ///     <c>char</c>, raised as <see cref="InvalidCastException"/>).</exception>
        /// <exception cref="IndexError">The coordinates of an empty axis (NumPy's <c>diffx[0]</c> of an empty
        ///     diff).</exception>
        /// <exception cref="FormatException">A string spacing that is not a number (.NET's parse; NumPy raises TypeError
        ///     for any str).</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: every slice view, coefficient array and edge temporary is released when the call
        ///     returns, and the output components are yielded to the caller through the returned
        ///     <see cref="np.GradientResult"/> (an <see cref="INDArrayCarrier"/>).
        /// </remarks>
        [NDScoped]
        internal static GradientResult Gradient(NDArray f, object[] varargs, int[] axisArg, int edge_order)
        {
            if (f is null) throw new ArgumentNullException(nameof(f));

            int N = f.ndim;

            // axes = normalize_axis_tuple(axis, N)  (all axes when axis is null). Done inline —
            // NumPy's normalize_axis_tuple REJECTS a repeated axis ("repeated axis"), while the
            // shared engine helper silently de-duplicates, so this matches NumPy exactly.
            int[] axes;
            if (axisArg is null)
            {
                axes = new int[N];
                for (int i = 0; i < N; i++) axes[i] = i;
            }
            else
            {
                axes = new int[axisArg.Length];
                for (int i = 0; i < axisArg.Length; i++)
                {
                    int ax = axisArg[i];
                    int adj = ax >= 0 ? ax : ax + N;        // reports the ORIGINAL axis on error
                    if (adj < 0 || adj >= N)
                        throw new AxisError(ax, N);
                    axes[i] = adj;
                }
                var seen = new HashSet<int>();
                foreach (int a in axes)
                    if (!seen.Add(a))
                        throw new ValueError("repeated axis");
            }
            int len_axes = axes.Length;

            // Resolve the spacing for each axis. Each entry is either a boxed double (uniform
            // spacing → weak coefficient) or a float64 1-D NDArray of successive differences
            // (non-uniform spacing → strong coefficients).
            int n = varargs.Length;
            object[] dx;
            if (n == 0)
            {
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++) dx[i] = 1.0;              // unit spacing
            }
            else if (n == 1 && GradSpacingNdim(varargs[0]) == 0)
            {
                double s = GradToDouble(varargs[0]);                          // one scalar for all axes
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++) dx[i] = s;
            }
            else if (n == len_axes)
            {
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++)
                {
                    object v = varargs[i];
                    if (GradSpacingNdim(v) == 0)
                    {
                        dx[i] = GradToDouble(v);                              // per-axis scalar
                        continue;
                    }
                    // NumPy's `distances = np.asanyarray(distances)`: a Python list is coerced by np.array's rules
                    // (Coordinates), where np.asanyarray took one level of an object[] and refused a ragged one with a
                    // KeyNotFoundException; GradSpacingNdim's walk has already raised NumPy's ValueError for a ragged one.
                    NDArray distances = Coordinates(v);
                    if (distances.ndim != 1)
                        throw new ValueError("distances must be either scalars or 1d");
                    if (distances.Shape.dimensions[0] != f.Shape.dimensions[axes[i]])
                        throw new ValueError(
                            "when 1d, distances must match the length of the corresponding dimension");
                    // Integer coordinates are converted to float64 to avoid modular arithmetic in diff.
                    if (IsIntegerLike(distances.GetTypeCode))
                        distances = distances.astype(np.float64);
                    NDArray diffx = np.diff(distances);
                    // Constant spacing collapses to the scalar (uniform) case — a real speedup and
                    // it matches NumPy bit-for-bit (the whole-array uniform formula runs instead).
                    // NumPy's own statement, `if (diffx == diffx[0]).all(): diffx = diffx[0]`: diffx[0] is NumPy's
                    // IndexError for the empty diff of an empty axis' coordinates (it read the empty buffer's first
                    // eight bytes), and the comparison runs in the diffs' own dtype, so complex diffs are constant only
                    // when both parts are (the double read compared the real parts).
                    NDArray first = diffx[0];
                    if (np.all(diffx == first))
                        dx[i] = SpacingElement(first, 0);
                    else
                        dx[i] = diffx;                                        // the diffs, in their own dtype
                }
            }
            else
            {
                throw new TypeError("invalid number of arguments");
            }

            if (edge_order > 2)
                throw new ValueError("'edge_order' greater than 2 not supported");

            // Output dtype (otype). Integer/char → compute in float64. Boolean → NumPy's subtract
            // TypeError. Float-family (incl. Decimal) keeps its dtype.
            NPTypeCode ftc = f.GetTypeCode;
            if (ftc == NPTypeCode.Boolean)
                throw new TypeError(
                    "numpy boolean subtract, the `-` operator, is not supported, " +
                    "use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
            NDArray work = IsIntegerLike(ftc) ? f.astype(np.float64) : f;
            DType otype = work.dtype;

            // Computation dtype for the NON-UNIFORM one-sided EDGES. NumPy forms those edges with
            // float64 coordinate coefficients (`ax_dx[i]` is a float64 scalar — STRONG under NEP50),
            // so a reduced-precision input is widened to result_type(float64, otype) == float64 for
            // the edge arithmetic and the quotient/combination is rounded ONCE to `otype` on store.
            // NumSharp treats a 0-D `NDArray.Scalar(double)` coefficient as a WEAK scalar, which would
            // demote the edge product/quotient back to float16/float32 and compute the edge in reduced
            // precision (observed 1–14 ULP off NumPy). Widening the edge slices to `edgeWiden`
            // reproduces NumPy's float64 edge exactly; float64/complex128/Decimal inputs need no
            // widening (the float64 scalar promotes up to them, and their edge already matches), so
            // `edgeWiden` is null there. The INTERIOR already computes in float64 — it multiplies by
            // 1-D float64 coefficient arrays, which NumSharp correctly keeps strong — so only the
            // edges need this. The UNIFORM path is unaffected: there NumPy's `2.*ax_dx` is a weak
            // Python float, so both libraries compute in the reduced dtype and already agree.
            DType edgeWiden = (otype == np.float32 || otype == np.float16) ? np.float64 : null;

            var outvals = new NDArray[len_axes];
            for (int ai = 0; ai < len_axes; ai++)
            {
                int axis = axes[ai];
                object ax_dx = dx[ai];
                long M = work.Shape.dimensions[axis];
                if (M < edge_order + 1)
                    throw new ValueError(
                        "Shape of array too small to calculate a numerical gradient, " +
                        "at least (edge_order + 1) elements are required.");

                bool uniform = ax_dx is double;
                // Text has no arithmetic loop: NumPy's first statement on it is the interior difference f[2:] - f[:-2]
                // (subtract of two string operands) or, for a non-uniform spacing, a * f[:-2] (float64 coefficients times
                // text). The checks above (spacing, shape) still come first, as in NumPy.
                if (Backends.Strings.NDStringUfuncs.IsVString(work))
                {
                    // StringDType: a new-style DType in the key, so NumPy's dispatcher names every operand's class.
                    string t = work.dtype.Meta.ToString();
                    throw uniform
                        ? Backends.Strings.NDStringUfuncs.VStringNoLoop("subtract", new[] { t, t })
                        : Backends.Strings.NDStringUfuncs.VStringNoLoop("multiply", new[] { np.float64.Meta.ToString(), t });
                }
                if (Backends.Strings.NDStringUfuncs.IsString(work))
                    throw uniform
                        ? Backends.Strings.NDStringUfuncs.NoLoop("subtract", work.dtype, work.dtype)
                        : Backends.Strings.NDStringUfuncs.NoLoop("multiply", np.float64, work.dtype);
                // NumPy's `out = np.empty_like(f, dtype=otype)`: the component keeps the input's memory order (order 'K':
                // an F-ordered input gives an F-ordered component, a broadcast or permuted one the order its strides sort
                // into), where a fresh C-ordered array of the dimensions was allocated. empty_like reads only the input's
                // dimensions and axis order, never its offset or a stride-0 axis, and the slice stores below write through
                // any layout.
                NDArray outp = np.empty_like(work, otype);

                // ---- interior: 2nd-order central differences over [1 : M-1] ----
                //   uniform     : out[1:-1] = (f[2:] - f[:-2]) / (2*dx)
                //   non-uniform : out[1:-1] = a*f[:-2] + b*f[1:-1] + c*f[2:]
                {
                    NDArray fm2 = SliceAlongAxis(work, axis, 0, M - 2);      // f[:-2]
                    NDArray fmid = SliceAlongAxis(work, axis, 1, M - 1);     // f[1:-1]
                    NDArray fp2 = SliceAlongAxis(work, axis, 2, M);          // f[2:]
                    if (uniform)
                    {
                        // Compute straight into the out view via ufunc out= — no intermediate
                        // full-size temp, no slice-copy. dtype:otype forces the loop to `otype`
                        // (so a float32 input divides in float32, matching NumPy's weak scalar).
                        NDArray ov = outp[BuildAxisSlices(N, axis, 1, M - 1)];
                        np.subtract(fp2, fm2, @out: ov, dtype: otype);       // ov = f[2:] - f[:-2]
                        using var half2 = NDArray.Scalar(2.0 * (double)ax_dx);
                        np.divide(ov, half2, @out: ov, dtype: otype);        // ov /= (2*dx)
                        ov.Dispose();
                    }
                    else
                    {
                        NDArray axd = (NDArray)ax_dx;                        // float64 diffs (len M-1)
                        NDArray dx1 = SliceAlongAxis(axd, 0, 0, M - 2);      // ax_dx[:-1]
                        NDArray dx2 = SliceAlongAxis(axd, 0, 1, M - 1);      // ax_dx[1:]
                        NDArray sum = dx1 + dx2;
                        // a = -dx2/(dx1*(dx1+dx2)); b = (dx2-dx1)/(dx1*dx2); c = dx1/(dx2*(dx1+dx2))
                        NDArray a = ReshapeAlong(-(dx2) / (dx1 * sum), axis, N);
                        NDArray b = ReshapeAlong((dx2 - dx1) / (dx1 * dx2), axis, N);
                        NDArray c = ReshapeAlong(dx1 / (dx2 * sum), axis, N);
                        // a*f[:-2] + b*f[1:-1] + c*f[2:] as ONE fused pass straight into the out view
                        // (same left-to-right elementwise order as NumPy, so bit-identical; the f64
                        // result is cast down to `otype` by the out= store for a float32/float16 input).
                        NDArray ov = outp[BuildAxisSlices(N, axis, 1, M - 1)];
                        var expr = NDExpr.Arr(a) * NDExpr.Arr(fm2)
                                 + NDExpr.Arr(b) * NDExpr.Arr(fmid)
                                 + NDExpr.Arr(c) * NDExpr.Arr(fp2);
                        np.evaluate(expr, @out: ov);
                        ov.Dispose();
                        dx1.Dispose(); dx2.Dispose(); sum.Dispose();
                        a.Dispose(); b.Dispose(); c.Dispose();
                    }
                    fm2.Dispose(); fmid.Dispose(); fp2.Dispose();
                }

                // ---- edges ----
                if (edge_order == 1)
                {
                    // out[0]  = (f[1]  - f[0])  / dx_0     (dx_0 = ax_dx    or ax_dx[0])
                    // out[-1] = (f[-1] - f[-2]) / dx_n     (dx_n = ax_dx    or ax_dx[-1])
                    // The coordinate diffs keep their dtype (float32 / float16 / bool / complex coordinates are not
                    // converted), so each end is read through that dtype (SpacingElement), never as a float64.
                    double dx0v = uniform ? (double)ax_dx : SpacingElement((NDArray)ax_dx, 0);
                    double dxnv = uniform ? (double)ax_dx : SpacingElement((NDArray)ax_dx, ((NDArray)ax_dx).size - 1);
                    WriteEdge(outp, work, axis, N, 0,
                        Edge1(SliceAlongAxis(work, axis, 1, 2), SliceAlongAxis(work, axis, 0, 1), dx0v, uniform, edgeWiden));
                    WriteEdge(outp, work, axis, N, M - 1,
                        Edge1(SliceAlongAxis(work, axis, M - 1, M), SliceAlongAxis(work, axis, M - 2, M - 1), dxnv, uniform, edgeWiden));
                }
                else
                {
                    // 2nd-order one-sided edges.
                    double a0, b0, c0, a1, b1, c1;
                    if (uniform)
                    {
                        double s = (double)ax_dx;
                        a0 = -1.5 / s; b0 = 2.0 / s; c0 = -0.5 / s;
                        a1 = 0.5 / s; b1 = -2.0 / s; c1 = 1.5 / s;
                    }
                    else
                    {
                        NDArray axd = (NDArray)ax_dx;
                        // Read through the diffs' own dtype (SpacingElement): see the first-order edges.
                        double d1 = SpacingElement(axd, 0), d2 = SpacingElement(axd, 1);
                        a0 = -(2.0 * d1 + d2) / (d1 * (d1 + d2));
                        b0 = (d1 + d2) / (d1 * d2);
                        c0 = -d1 / (d2 * (d1 + d2));
                        long L = axd.size;
                        double e1 = SpacingElement(axd, L - 2), e2 = SpacingElement(axd, L - 1);
                        a1 = e2 / (e1 * (e1 + e2));
                        b1 = -(e2 + e1) / (e1 * e2);
                        c1 = (2.0 * e2 + e1) / (e2 * (e1 + e2));
                    }
                    // front: out[0] = a0*f[0] + b0*f[1] + c0*f[2]
                    WriteEdge(outp, work, axis, N, 0,
                        Edge2(SliceAlongAxis(work, axis, 0, 1), SliceAlongAxis(work, axis, 1, 2),
                              SliceAlongAxis(work, axis, 2, 3), a0, b0, c0, uniform, edgeWiden));
                    // back: out[-1] = a1*f[-3] + b1*f[-2] + c1*f[-1]
                    WriteEdge(outp, work, axis, N, M - 1,
                        Edge2(SliceAlongAxis(work, axis, M - 3, M - 2), SliceAlongAxis(work, axis, M - 2, M - 1),
                              SliceAlongAxis(work, axis, M - 1, M), a1, b1, c1, uniform, edgeWiden));
                }

                outvals[ai] = outp;
            }

            return new GradientResult(outvals, single: len_axes == 1);
        }

        /// <summary>
        ///     One first-order gradient edge, <c>out[0]</c> or <c>out[-1]</c> = <c>(hi - lo) / d</c>. Uniform spacing
        ///     uses the weak C# double <paramref name="d"/>, so the quotient stays in the input dtype (NumPy's weak
        ///     <c>ax_dx</c>); non-uniform spacing divides by the STRONG float64 coordinate scalar.
        /// </summary>
        /// <param name="hi">The later length-1 slice along the axis (<c>f[1]</c> or <c>f[-1]</c>).</param>
        /// <param name="lo">The earlier length-1 slice along the axis (<c>f[0]</c> or <c>f[-2]</c>).</param>
        /// <param name="d">The spacing to divide by.</param>
        /// <param name="uniform">True for scalar spacing (weak double divisor); false for coordinate spacing
        /// (strong float64 divisor).</param>
        /// <param name="widenTo">float64 for a reduced-precision float16/float32 input, else null. The difference
        /// is widened to it BEFORE the divide, so the quotient is computed in float64 and rounded once to otype on
        /// store, reproducing NumPy; a weak 0-D <c>NDArray.Scalar(double)</c> would demote it back to the input
        /// precision. The subtraction itself runs in the input dtype first (as NumPy's <c>f[s2]-f[s3]</c> does);
        /// widening is exact, so the order cannot change the result.</param>
        /// <returns>The edge values, length 1 along the axis.</returns>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the difference, the widened copy and the 0-d divisor are released when
        ///     the edge returns, and the result is yielded to the caller. <paramref name="hi"/> and
        ///     <paramref name="lo"/> are views the caller built before this scope opened, so this method no longer
        ///     disposes them: <c>Gradient</c>'s own scope releases them.
        ///     <para>This helper keeps a scope of its own instead of <c>[NDScopedCovered]</c>, although its only
        ///     caller is scoped. An edge slice holds one hyperplane of the input, which is as large as the whole
        ///     array divided by the axis length: along a short axis (length 3, say) each temporary here is a third
        ///     of the input. <c>Gradient</c>'s scope would hold them until every axis is done. The nested scope
        ///     costs one scope allocation per call (about 220 bytes), two calls per gradient axis.</para>
        /// </remarks>
        [NDScoped]
        private static NDArray Edge1(NDArray hi, NDArray lo, double d, bool uniform, DType widenTo)
        {
            NDArray num = hi - lo;
            NDArray res;
            if (uniform)
            {
                res = num / d;
            }
            else if (widenTo is null)
            {
                // float64 / complex128 / Decimal input: the float64 scalar promotes up to the operand's
                // dtype, so the edge already computes at full precision — keep the existing path.
                var sd = NDArray.Scalar(d);
                res = num / sd;
            }
            else
            {
                NDArray w = num.astype(widenTo);   // exact widen f16/f32 -> f64
                var sd = NDArray.Scalar(d);        // f64 / f64 -> f64 (no weak demotion)
                res = w / sd;
            }
            return res;
        }

        /// <summary>
        ///     One second-order gradient edge: <c>a*f0 + b*f1 + c*f2</c>, left-associative exactly as NumPy evaluates
        ///     it. Uniform spacing uses the weak C# double coefficients, so the combination stays in the input dtype
        ///     (NumPy's weak coefficients); non-uniform spacing uses STRONG float64 coordinate coefficients.
        /// </summary>
        /// <param name="f0">The first length-1 slice of the three-point stencil.</param>
        /// <param name="f1">The second length-1 slice.</param>
        /// <param name="f2">The third length-1 slice.</param>
        /// <param name="a">The coefficient of <paramref name="f0"/>.</param>
        /// <param name="b">The coefficient of <paramref name="f1"/>.</param>
        /// <param name="c">The coefficient of <paramref name="f2"/>.</param>
        /// <param name="uniform">True for scalar spacing (weak double coefficients); false for coordinate spacing
        /// (strong float64 coefficients).</param>
        /// <param name="widenTo">float64 for a reduced-precision float16/float32 input, else null. The three slices
        /// are widened to it BEFORE the products, so <c>a*f0</c> and the rest are computed in float64 (NumPy widens
        /// the f32 operand into the float64 multiply) and the float64 sum is rounded once to otype on store; a weak
        /// 0-D <c>NDArray.Scalar(double)</c> coefficient would instead demote each product back to the input
        /// precision and diverge by several ULP.</param>
        /// <returns>The edge values, length 1 along the axis.</returns>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the 0-d coefficients, the widened slices, the three products and the
        ///     partial sum are released when the edge returns, and the result is yielded to the caller. The three
        ///     slices are views the caller built before this scope opened, so this method no longer disposes them:
        ///     <c>Gradient</c>'s own scope releases them.
        ///     <para>This helper keeps a scope of its own instead of <c>[NDScopedCovered]</c> for the reason given
        ///     on <see cref="Edge1"/>: the widened slices, the products and the partial sum are each one hyperplane
        ///     of the input, and the caller's scope would hold them until every axis is done.</para>
        /// </remarks>
        [NDScoped]
        private static NDArray Edge2(NDArray f0, NDArray f1, NDArray f2, double a, double b, double c, bool uniform, DType widenTo)
        {
            NDArray res;
            if (uniform)
            {
                NDArray t0 = f0 * a, t1 = f1 * b, t2 = f2 * c;
                NDArray s01 = t0 + t1; res = s01 + t2;
            }
            else if (widenTo is null)
            {
                // float64 / complex128 / Decimal input: the float64 scalar coefficients promote up to the
                // operand dtype, so the edge already computes at full precision — keep the existing path.
                NDArray na = NDArray.Scalar(a), nb = NDArray.Scalar(b), nc = NDArray.Scalar(c);
                NDArray t0 = na * f0, t1 = nb * f1, t2 = nc * f2;
                NDArray s01 = t0 + t1; res = s01 + t2;
            }
            else
            {
                // reduced precision: widen the slices to float64 so the products and sum run in float64
                // (the f64 scalar coefficients no longer demote against an f32/f16 operand), then the
                // caller's store rounds the float64 combination once to otype — bit-identical to NumPy.
                NDArray w0 = f0.astype(widenTo), w1 = f1.astype(widenTo), w2 = f2.astype(widenTo);
                NDArray na = NDArray.Scalar(a), nb = NDArray.Scalar(b), nc = NDArray.Scalar(c);
                NDArray t0 = na * w0, t1 = nb * w1, t2 = nc * w2;
                NDArray s01 = t0 + t1; res = s01 + t2;
            }
            return res;
        }

        // Assign a length-1 edge slice at index `at` along `axis` into `outp`, then dispose the value.
        /// <summary>
        ///     Stores one computed edge into the output — NumPy's <c>out[tuple(slice1)] = ...</c> with <c>slice1[axis]</c> the
        ///     edge index — and releases the computed value at once (one hyperplane of the input, see <see cref="Edge1"/>).
        /// </summary>
        /// <param name="outp">The output component being filled.</param>
        /// <param name="work">The (converted) input; unused by the store, kept for the call sites' symmetry.</param>
        /// <param name="axis">The axis the edge lies on.</param>
        /// <param name="N">The rank of <paramref name="outp"/>.</param>
        /// <param name="at">The edge's index along <paramref name="axis"/> (0 or the last).</param>
        /// <param name="value">The edge values, length 1 along <paramref name="axis"/>; disposed here.</param>
        /// <exception cref="ValueError">A value that does not broadcast into the length-1 slot (an edge slice that came out
        ///     empty: NumPy's integer index raises IndexError there instead).</exception>
        private static void WriteEdge(NDArray outp, NDArray work, int axis, int N, long at, NDArray value)
        {
            outp[BuildAxisSlices(N, axis, at, at + 1)] = value;
            value.Dispose();
        }

        // Reshape a 1-D coefficient array to broadcast along `axis`: shape is all-1 except `axis`.
        /// <summary>
        ///     NumPy's <c>a.reshape(shape)</c> with <c>shape = np.ones(N); shape[axis] = -1</c>: a 1-D coefficient array laid
        ///     along <paramref name="axis"/> so it broadcasts over the other axes of the input.
        /// </summary>
        /// <param name="a">A 1-D coefficient array; disposed here (the reshaped view keeps its buffer alive).</param>
        /// <param name="axis">The axis the coefficients run along.</param>
        /// <param name="N">The input's rank.</param>
        /// <returns>A view of <paramref name="a"/>'s buffer with shape all-1 except <paramref name="axis"/>.</returns>
        private static NDArray ReshapeAlong(NDArray a, int axis, int N)
        {
            long[] shp = new long[N];
            for (int i = 0; i < N; i++) shp[i] = 1;
            shp[axis] = a.Shape.dimensions[0];
            var r = a.reshape(Shape.Adopt(shp));
            a.Dispose();
            return r;
        }

        // Slice spec [:, ..., start:stop, ..., :] with a length-(stop-start) range on `axis`.
        /// <summary>
        ///     NumPy's <c>tuple(slice1)</c>: every axis taken whole except <paramref name="axis"/>, which takes
        ///     <c>start:stop</c>.
        /// </summary>
        /// <param name="N">The rank.</param>
        /// <param name="axis">The sliced axis.</param>
        /// <param name="start">The first index along <paramref name="axis"/>.</param>
        /// <param name="stop">One past the last index.</param>
        /// <returns>A fresh slice list for the indexer.</returns>
        private static Slice[] BuildAxisSlices(int N, int axis, long start, long stop)
        {
            var slices = new Slice[N];
            for (int i = 0; i < N; i++)
                slices[i] = i == axis ? new Slice(start, stop) : Slice.All;
            return slices;
        }

        // ndim of a spacing argument: 0 for a C# scalar, else the NDArray's ndim.
        /// <summary>
        ///     NumPy's <c>np.ndim(v)</c> for a spacing (<c>np.asarray(v).ndim</c>), which decides whether it is a step (0) or
        ///     coordinates: 0 for a C# scalar or a string, the rank of an NDArray or a typed C# array, 1 for a
        ///     <see cref="Memory{T}"/> of a dtype, and the nesting depth of a Python sequence (a <c>List&lt;T&gt;</c>, an
        ///     <c>object[]</c>, a tuple: the house map, <see cref="PolySequence.IsSequence"/>).
        /// </summary>
        /// <param name="v">The spacing.</param>
        /// <returns>Its rank.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="v"/> is null (NumPy's <c>np.ndim(None)</c> is 0, and its
        ///     arithmetic then raises TypeError).</exception>
        /// <exception cref="ValueError">A ragged sequence (np.asarray's inhomogeneous-shape text, raised where NumPy's
        ///     np.ndim raises it).</exception>
        /// <remarks>
        ///     A sequence was read as a scalar (only a C# array was a coordinate list), so a <c>List&lt;double&gt;</c> of
        ///     coordinates reached the double conversion and its InvalidCastException, and an <c>object[]</c> was measured
        ///     by its C# rank, not by what it holds.
        /// </remarks>
        private static int GradSpacingNdim(object v)
        {
            if (v is null) throw new ArgumentNullException(nameof(v));
            if (v is NDArray nd) return nd.ndim;
            // A Python sequence is tested before a C# array: an object[] (and any array whose elements are no dtype) is a
            // list, whose rank is the depth np.array discovers in it, not the C# array's own rank.
            if (PolySequence.IsSequence(v)) return PolySequence.NDim(v);
            if (v is Array arr) return arr.Rank;    // a typed C# array is an ndarray of its rank
            return PolySequence.IsArrayLike(v) ? 1 : 0;    // a Memory<T> of a dtype is 1-D; anything else a scalar
        }

        /// <summary>
        ///     NumPy's <c>np.asanyarray(distances)</c> for a coordinate spacing: an NDArray as it is, a Python sequence
        ///     coerced by np.array's rules (<see cref="PolySequence.ToArray"/>: a Python int item is int64, a float float64,
        ///     nested lists any depth), a typed C# array or a <see cref="Memory{T}"/> as a whole.
        /// </summary>
        /// <param name="v">A spacing whose <see cref="GradSpacingNdim"/> is not 0.</param>
        /// <returns>The coordinates as an array (the caller checks its rank and length).</returns>
        /// <exception cref="NotSupportedException">A sequence holding None or another object, or a Python int past
        ///     uint64: NumPy builds an object array, a dtype NumSharp does not have.</exception>
        /// <remarks>
        ///     A list of str is NumPy's str array, which NumSharp holds: it is converted by <c>np.asanyarray</c>, and the
        ///     differences of its coordinates raise NumPy's subtract TypeError as for a str NDArray.
        /// </remarks>
        private static NDArray Coordinates(object v)
        {
            if (v is NDArray nd)
                return nd;
            if (PolySequence.IsSequence(v))
            {
                NDArray coords = PolySequence.ToArrayOrNonNumeric(v, out PolyNonNumericArray nonNumeric);
                if (coords is not null)
                    return coords;
                // The polynomial coercion reports a str array as a shape (it computes no str arithmetic); np.gradient's
                // next statement, np.diff, refuses one by NumPy's own text, so the str array is built. An object array has
                // no NumSharp dtype at all.
                if (nonNumeric.IsObject)
                    throw nonNumeric.Refusal;
            }
            return np.asanyarray(v);
        }

        // Convert a scalar spacing to double. A 0-d array spacing is read through its own dtype (SpacingElement): a 0-d
        // int64 2 read as a float64 is the denormal 1e-323, and a 0-d float32 read as a float64 reaches past its buffer.
        /// <summary>A scalar spacing (a C# number or string, or a 0-d array) as the double step the stencil uses.</summary>
        /// <param name="v">The spacing (not null; <see cref="GradSpacingNdim"/> has refused null).</param>
        /// <returns>The step.</returns>
        /// <exception cref="InvalidCastException">A value <see cref="Convert.ToDouble(object)"/> refuses (a
        ///     <see cref="Half"/>, a <c>char</c>).</exception>
        /// <exception cref="FormatException">A string that is not a number.</exception>
        private static double GradToDouble(object v)
        {
            if (v is NDArray nd) return SpacingElement(nd, 0);
            return Convert.ToDouble(v);
        }

        /// <summary>
        ///     One element of a spacing array (a 0-d spacing, or the diffs of a coordinate array) as a double, read through
        ///     the array's own dtype.
        /// </summary>
        /// <param name="a">The array: any dtype np.diff or the caller gives (float64, float32, float16, bool, complex, ...).</param>
        /// <param name="index">The flat C-order index of the element (0 for a 0-d array).</param>
        /// <returns>The element converted to double (a complex element's real part, a bool's 1 or 0).</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is outside the array.</exception>
        /// <remarks>
        ///     <c>NDArray.GetDouble</c> is not a conversion: it indexes the float64 lane of the storage's typed-slice union
        ///     whatever the array's dtype. On a float32 diff array that reads two floats' bits as one double and, from the
        ///     middle of the array on, eight bytes per element past its end, so np.gradient's results over float32, float16
        ///     or bool coordinates changed from run to run; on an int64 element it reads the integer's bits. Reading the
        ///     element boxed by its own dtype (<c>GetAtIndex</c>, which also maps the index through a view's offset and
        ///     strides) and converting it is exact for every dtype.
        /// </remarks>
        private static double SpacingElement(NDArray a, long index) => Converts.ToDouble(a.GetAtIndex(index));

        /// <summary>NumPy's <c>np.issubdtype(dtype, np.integer)</c> over NumSharp's dtypes: the integer dtypes and
        ///     <c>char</c> (NumPy's uint16), not bool.</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>Whether the input or the coordinates convert to float64 before the differences.</returns>
        private static bool IsIntegerLike(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
            or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
            or NPTypeCode.Char => true,
            _ => false,
        };
    }
}
