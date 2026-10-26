using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
// PolyNumber's operator kinds (BinaryOp), its Python values (PyScalar) and its promotion table (PolyTyping).
using NumSharp.Backends.Kernels;
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
    // Every statement computes in NumPy's dtype, so the result is bit-identical to
    // NumPy's:
    //   * Output dtype (`otype`): a float-family input keeps its dtype
    //     (Half/Single/Double/Complex; NumSharp-only Decimal too); an
    //     integer/char input is promoted to float64 (NumPy converts f to float64
    //     first, matching np.diff-free arithmetic); a BOOLEAN input raises the
    //     same "numpy boolean subtract" TypeError NumPy raises (bool is not an
    //     integer subtype, so NumPy leaves it bool and the internal subtract
    //     fails).
    //   * Spacing kinds: a scalar spacing stays the value NumPy's `dx` list holds,
    //     and each scalar statement (`2. * ax_dx`, the edge coefficients, the
    //     quotients and products of the edges) runs in NEP 50's loop dtype for its
    //     operands (PolyNumber, NumPy's three arithmetics). A Python number (a C#
    //     int / double / bool / Complex / BigInteger) is WEAK, so a float32 input
    //     stays float32; a NumPy scalar (Half, char, decimal), a 0-d array and a
    //     collapsed coordinate step (`diffx[0]`) are STRONG, so a float64 step
    //     computes a float32 input in float64, rounded once into the component.
    //     Coordinate differences keep the coordinates' dtype (only integer
    //     coordinates are converted to float64), and so do the coefficients built
    //     from them: float32 coordinates give float32 coefficients. A complex
    //     result written into a real component keeps its real part (NumPy's
    //     setitem). The port converted every scalar step to a C# double (a weak
    //     Python float) and widened the non-uniform edges to float64, on the belief
    //     that NumPy converts all coordinates to float64.
    //
    // complex128: the divisions are NumPy's CDOUBLE_divide (Smith's algorithm, the
    // house kernel), bit-exact for finite values. A NaN's sign can still differ for
    // special operands, where NumPy's complex divide keeps another operand's NaN in
    // one part: the library residual of the binary complex kernels (pinned by
    // OpenBugs.NaNOperandOrder's Complex128_AddMultiplyDivide_KeepNumPysNaNPerPart),
    // which gradient inherits.

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
        /// <exception cref="ArgumentNullException"><paramref name="f"/> is null.</exception>
        /// <exception cref="AxisError">An axis outside <c>[-f.ndim, f.ndim)</c>.</exception>
        /// <exception cref="ValueError">A repeated axis; a ragged list spacing (np.ndim's inhomogeneous-shape text); a
        ///     coordinate array of another rank than 1 or another length than its axis; <paramref name="edge_order"/> above
        ///     2; an axis shorter than <c>edge_order + 1</c>.</exception>
        /// <exception cref="NotSupportedException">A list spacing holding None or another object, or a Python int past
        ///     uint64 (NumPy's object array, a dtype NumSharp does not have).</exception>
        /// <exception cref="TypeError">A spacing count other than 0, 1 or the number of axes ("invalid number of
        ///     arguments"); a bool <paramref name="f"/> (NumPy's boolean subtract); a text <paramref name="f"/> (NumPy's
        ///     no-loop texts of the subtract or the coefficient multiply); bool coordinates (the boolean negative of the
        ///     coefficients); a str or None step, refused by Python's <c>2. * ax_dx</c> (<c>can't multiply sequence by
        ///     non-int of type 'float'</c>, <c>unsupported operand type(s) for *: 'float' and 'NoneType'</c>).</exception>
        /// <exception cref="IndexError">The coordinates of an empty axis (NumPy's <c>diffx[0]</c> of an empty
        ///     diff).</exception>
        /// <exception cref="DivideByZeroException">A zero step that is a Python number with second-order edges (CPython's
        ///     ZeroDivisionError of <c>-1.5 / ax_dx</c>: <c>float division by zero</c>, or <c>complex division by zero</c>
        ///     for a complex step; a NumPy-scalar or 0-d array zero step gives inf / nan instead, as in NumPy); a decimal
        ///     input or step divided by zero (decimal has no inf).</exception>
        /// <exception cref="OverflowException">A Python int step past the float range (a <c>BigInteger</c>: CPython's
        ///     <c>int too large to convert to float</c> in <c>2. * ax_dx</c>); a decimal result past decimal's
        ///     range.</exception>
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

            // Resolve the spacing for each axis, as NumPy's `dx` list holds it. A scalar spacing stays the object it is
            // (SpacingOf): a Python number (weak under NEP 50), a NumPy scalar or 0-d array (strong, of its own dtype), or
            // a str / None, which NumPy accepts here and refuses only in `2. * ax_dx`. A coordinate array becomes the
            // differences of its coordinates in their own dtype (integers converted to float64), or the NumPy scalar
            // `diffx[0]` when every difference equals the first. Converting a scalar to a double here (the port did) made a
            // NumPy scalar weak, refused a Half or a char, and parsed a str.
            int n = varargs.Length;
            object[] dx;
            if (n == 0)
            {
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++) dx[i] = UnitSpacing;        // NumPy's [1.0] * len_axes: a Python float
            }
            else if (n == 1 && GradSpacingNdim(varargs[0]) == 0)
            {
                object s = SpacingOf(varargs[0]);                              // varargs * len_axes: one object, every axis
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
                        dx[i] = SpacingOf(v);                                  // per-axis scalar, kept as it is
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
                        dx[i] = PolyNumber.ScalarOf(first);                   // diffx[0]: a NumPy scalar (strong)
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
            // The scalar statements below (`2. * ax_dx`, the edge coefficients, the edge quotients and products) run
            // through PolyNumber, NumPy's three arithmetics by operand kind: every operation's loop dtype is NEP 50's for
            // its operands, so a weak Python step keeps a float32 input in float32 and a NumPy scalar of float64 (a
            // collapsed float64 coordinate step, a 0-d float64 array) computes in float64, rounded once into the
            // component. The port widened the non-uniform edges to float64 instead (edgeWiden), on the belief that NumPy
            // converts every coordinate array to float64; it converts only integer coordinates.

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

                // NumPy's `uniform_spacing = np.ndim(ax_dx) == 0`: every scalar spacing (a PolyNumber, a str, a None) is
                // uniform; only a coordinate array's differences (an NDArray of rank 1) are not.
                bool uniform = ax_dx is not NDArray;
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
                        // NumPy's `out[s1] = (f[s4] - f[s2]) / (2. * ax_dx)`, computed straight into the out view via
                        // ufunc out= (no full-size temporary): the difference in the input's dtype first (Python evaluates
                        // `/`'s left operand first, so a str or None step raises after it), then `2. * ax_dx` in the step's
                        // own arithmetic (TwiceSpacing), then the quotient in NEP 50's loop dtype for the difference and that
                        // value: the input's dtype for a weak Python number (a float32 input divides in float32), the
                        // promotion for a NumPy scalar (a float32 input divided by a float64 step divides in float64, and
                        // the out= store rounds once).
                        NDArray ov = outp[BuildAxisSlices(N, axis, 1, M - 1)];
                        np.subtract(fp2, fm2, @out: ov, dtype: otype);       // ov = f[2:] - f[:-2]
                        PolyNumber twice = TwiceSpacing(ax_dx);
                        NPTypeCode loop = PolyNumber.LoopDtype(BinaryOp.Divide, PolyNumber.FromArray(ov), twice);
                        if (StoresInPlace(loop, otype))
                        {
                            using var divisor = twice.AsArrayOf(loop);
                            np.divide(ov, divisor, @out: ov, dtype: loop);    // ov /= (2*dx)
                        }
                        else
                            StoreComponent(ov, PolyNumber.Binary(BinaryOp.Divide, PolyNumber.FromArray(ov), twice).Array);
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
                        // The fused pass types every node as NumPy types each ufunc call: the coefficients carry the
                        // coordinates' dtype (float32 coordinates give float32 coefficients). A complex combination (complex
                        // coordinates) into a real component is NumPy's setitem, which keeps the real part, where the fused
                        // pass's out= refuses the cast: it is evaluated into a temporary and stored.
                        if (StoresInPlace(PolyTyping.Promote(a.typecode, work.typecode), otype))
                            np.evaluate(expr, @out: ov);
                        else
                            StoreComponent(ov, np.evaluate(expr));
                        ov.Dispose();
                        dx1.Dispose(); dx2.Dispose(); sum.Dispose();
                        a.Dispose(); b.Dispose(); c.Dispose();
                    }
                    fm2.Dispose(); fmid.Dispose(); fp2.Dispose();
                }

                // ---- edges ----
                // NumPy indexes the edges with INTEGERS (`slice1[axis] = 0`, `slice2[axis] = 1`, ...): an edge value is
                // the (N-1)-D hyperplane f[..., k, ...], a NumPy SCALAR for a 1-D input, so a 1-D input's edges are
                // scalarmath (a sum of two NaNs keeps the first operand's, a complex product is the naive one) where an
                // N-D input's are ufuncs; and an index past a short axis is NumPy's IndexError, which only the
                // second-order branch reaches (edge_order 0 / -1 admit axes of 1 / 0 elements). The port took length-1
                // slices, which never raise and always broadcast as arrays.
                if (edge_order == 1)
                {
                    // out[0]  = (f[1]  - f[0])  / dx_0     (dx_0 = ax_dx    or ax_dx[0])
                    // out[-1] = (f[-1] - f[-2]) / dx_n     (dx_n = ax_dx    or ax_dx[-1])
                    // A non-uniform step is NumPy's `ax_dx[0]` / `ax_dx[-1]`: a NumPy scalar of the differences' dtype.
                    PolyNumber dx0 = uniform ? (PolyNumber)ax_dx : PolyNumber.ScalarOf(((NDArray)ax_dx)[0]);
                    WriteEdge(outp, work, axis, N, 0, Edge1(work, axis, N, 1, 0, dx0));
                    PolyNumber dxn = uniform ? (PolyNumber)ax_dx : PolyNumber.ScalarOf(((NDArray)ax_dx)[-1]);
                    WriteEdge(outp, work, axis, N, -1, Edge1(work, axis, N, -1, -2, dxn));
                }
                else
                {
                    // 2nd-order one-sided edges, NumPy's statements in its order: the front coefficients, the front edge,
                    // the back coefficients, the back edge. A uniform step's coefficients are a Python float divided by the
                    // step (CPython's ZeroDivisionError for a Python 0.0, scalarmath inf / nan for a NumPy zero); a
                    // non-uniform step's are scalarmath over the NumPy scalars ax_dx[0], ax_dx[1] (front) and ax_dx[-2],
                    // ax_dx[-1] (back), in the differences' dtype.
                    PolyNumber a0, b0, c0;
                    if (uniform)
                    {
                        var s = (PolyNumber)ax_dx;
                        a0 = Div(Py(-1.5), s); b0 = Div(Py(2.0), s); c0 = Div(Py(-0.5), s);
                    }
                    else
                    {
                        // a = -(2. * dx1 + dx2) / (dx1 * (dx1 + dx2)); b = (dx1 + dx2) / (dx1 * dx2);
                        // c = - dx1 / (dx2 * (dx1 + dx2))
                        NDArray axd = (NDArray)ax_dx;
                        PolyNumber d1 = PolyNumber.ScalarOf(axd[0]), d2 = PolyNumber.ScalarOf(axd[1]);
                        a0 = Div(PolyNumber.Negate(Add(Mul(Py(2.0), d1), d2)), Mul(d1, Add(d1, d2)));
                        b0 = Div(Add(d1, d2), Mul(d1, d2));
                        c0 = Div(PolyNumber.Negate(d1), Mul(d2, Add(d1, d2)));
                    }
                    // front: out[0] = a0*f[0] + b0*f[1] + c0*f[2]; f[1] / f[2] of an axis of 1 / 2 elements is the
                    // IndexError, raised after the coefficients (a Python zero step's ZeroDivisionError comes first).
                    WriteEdge(outp, work, axis, N, 0, Edge2(work, axis, N, 0, 1, 2, a0, b0, c0));
                    PolyNumber a1, b1, c1;
                    if (uniform)
                    {
                        var s = (PolyNumber)ax_dx;
                        a1 = Div(Py(0.5), s); b1 = Div(Py(-2.0), s); c1 = Div(Py(1.5), s);
                    }
                    else
                    {
                        // a = (dx2) / (dx1 * (dx1 + dx2)); b = - (dx2 + dx1) / (dx1 * dx2);
                        // c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
                        NDArray axd = (NDArray)ax_dx;
                        PolyNumber e1 = PolyNumber.ScalarOf(axd[-2]), e2 = PolyNumber.ScalarOf(axd[-1]);
                        a1 = Div(e2, Mul(e1, Add(e1, e2)));
                        b1 = Div(PolyNumber.Negate(Add(e2, e1)), Mul(e1, e2));
                        c1 = Div(Add(Mul(Py(2.0), e2), e1), Mul(e2, Add(e1, e2)));
                    }
                    // back: out[-1] = a1*f[-3] + b1*f[-2] + c1*f[-1] (an axis short enough to make these raise has
                    // already raised at the front edge)
                    WriteEdge(outp, work, axis, N, -1, Edge2(work, axis, N, -3, -2, -1, a1, b1, c1));
                }

                outvals[ai] = outp;
            }

            return new GradientResult(outvals, single: len_axes == 1);
        }

        /// <summary>
        ///     One first-order gradient edge, NumPy's <c>(f[s2] - f[s3]) / dx</c> with integer indices along the axis: the
        ///     difference of two edge values (<see cref="EdgeOperand"/>: NumPy scalars for a 1-D input, (N-1)-D arrays
        ///     otherwise) in the input's dtype, then the quotient in NEP 50's loop dtype for the step's own kind.
        /// </summary>
        /// <param name="f">The (converted) input.</param>
        /// <param name="axis">The axis the edge lies on.</param>
        /// <param name="N">The input's rank.</param>
        /// <param name="kHi">The later index along the axis (<c>1</c> or <c>-1</c>), indexed first as Python does.</param>
        /// <param name="kLo">The earlier index along the axis (<c>0</c> or <c>-2</c>).</param>
        /// <param name="d">The step: the uniform spacing as NumPy holds it (a weak Python number keeps the input's dtype; a
        ///     NumPy scalar or 0-d array promotes with it), or a non-uniform spacing's <c>ax_dx[0]</c> / <c>ax_dx[-1]</c>, a
        ///     NumPy scalar of the coordinate differences' dtype.</param>
        /// <returns>The edge value: an (N-1)-D array, or a 0-d array holding the NumPy scalar of a 1-D input.</returns>
        /// <exception cref="TypeError">A bool or text input (NumPy's subtract refusals).</exception>
        /// <exception cref="IndexError">An index outside the axis (unreachable for edge_order 1, whose axes hold at least two
        ///     elements).</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the two edge views, the difference and the converted operands of the quotient are
        ///     released when the edge returns, and the result is yielded to the caller.
        ///     <para>This helper keeps a scope of its own instead of <c>[NDScopedCovered]</c>, although its only
        ///     caller is scoped. An edge slice holds one hyperplane of the input, which is as large as the whole
        ///     array divided by the axis length: along a short axis (length 3, say) each temporary here is a third
        ///     of the input. <c>Gradient</c>'s scope would hold them until every axis is done. The nested scope
        ///     costs one scope allocation per call (about 220 bytes), two calls per gradient axis.</para>
        /// </remarks>
        [NDScoped]
        private static NDArray Edge1(NDArray f, int axis, int N, long kHi, long kLo, PolyNumber d)
        {
            PolyNumber hi = EdgeOperand(AtAxis(f, axis, N, kHi));
            PolyNumber lo = EdgeOperand(AtAxis(f, axis, N, kLo));
            PolyNumber num = PolyNumber.Binary(BinaryOp.Subtract, hi, lo);
            return PolyNumber.Binary(BinaryOp.Divide, num, d).ToNDArray();
        }

        /// <summary>
        ///     One second-order gradient edge, NumPy's <c>a * f[s2] + b * f[s3] + c * f[s4]</c> with integer indices along the
        ///     axis: evaluated left to right as Python does, each edge value indexed just before its product (so an index past
        ///     a short axis raises after the products before it), each product and sum in NEP 50's loop dtype for its operands
        ///     (a weak Python coefficient keeps the input's dtype; a NumPy scalar coefficient promotes with it, so a float64
        ///     coordinate coefficient computes a float32 input's edge in float64), the coefficient first in each product, as
        ///     NumPy writes it. A 1-D input's edge values are NumPy scalars, so its products and sums are scalarmath.
        /// </summary>
        /// <param name="f">The (converted) input.</param>
        /// <param name="axis">The axis the edge lies on.</param>
        /// <param name="N">The input's rank.</param>
        /// <param name="k0">The index of the first stencil point (<c>0</c> or <c>-3</c>).</param>
        /// <param name="k1">The index of the second (<c>1</c> or <c>-2</c>).</param>
        /// <param name="k2">The index of the third (<c>2</c> or <c>-1</c>).</param>
        /// <param name="a">The coefficient of the first point (a Python number or a NumPy scalar).</param>
        /// <param name="b">The coefficient of the second.</param>
        /// <param name="c">The coefficient of the third.</param>
        /// <returns>The edge value: an (N-1)-D array, or a 0-d array holding the NumPy scalar of a 1-D input.</returns>
        /// <exception cref="TypeError">A text input (NumPy's multiply refusal).</exception>
        /// <exception cref="IndexError">An index outside an axis of one or two elements (edge_order 0 or -1 admits them):
        ///     <c>index 2 is out of bounds for axis 0 with size 2</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the edge views, the converted operands, the three products and the partial sum are
        ///     released when the edge returns, and the result is yielded to the caller.
        ///     <para>This helper keeps a scope of its own instead of <c>[NDScopedCovered]</c> for the reason given on
        ///     <see cref="Edge1"/>: the products and the partial sum are each one hyperplane of the input, and the caller's
        ///     scope would hold them until every axis is done.</para>
        /// </remarks>
        [NDScoped]
        private static NDArray Edge2(NDArray f, int axis, int N, long k0, long k1, long k2,
                                     PolyNumber a, PolyNumber b, PolyNumber c)
        {
            PolyNumber t0 = Mul(a, EdgeOperand(AtAxis(f, axis, N, k0)));
            PolyNumber t1 = Mul(b, EdgeOperand(AtAxis(f, axis, N, k1)));
            PolyNumber s01 = Add(t0, t1);
            PolyNumber t2 = Mul(c, EdgeOperand(AtAxis(f, axis, N, k2)));
            return Add(s01, t2).ToNDArray();
        }

        /// <summary>
        ///     NumPy's <c>f[tuple(slice)]</c> with an integer at <paramref name="axis"/>: the hyperplane at index
        ///     <paramref name="k"/>, one rank lower than <paramref name="a"/> (a 0-d view of a 1-D array).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The indexed axis.</param>
        /// <param name="N">The array's rank.</param>
        /// <param name="k">The index (negative counts from the end, as in Python).</param>
        /// <returns>A view.</returns>
        /// <exception cref="IndexError"><paramref name="k"/> is outside the axis: NumPy's <c>index {k} is out of bounds for
        ///     axis {axis} with size {M}</c>, the index as given.</exception>
        private static NDArray AtAxis(NDArray a, int axis, int N, long k) => a[BuildAxisIndex(N, axis, k)];

        /// <summary>
        ///     An edge value as Python holds it: integer indexing of a 1-D array gives a NumPy scalar (scalarmath from then
        ///     on), of an N-D array an (N-1)-D array (ufuncs).
        /// </summary>
        /// <param name="v">The indexed hyperplane (<see cref="AtAxis"/>).</param>
        /// <returns>A NumPy scalar for a 0-d view, else the array.</returns>
        private static PolyNumber EdgeOperand(NDArray v) => v.ndim == 0 ? PolyNumber.ScalarOf(v) : PolyNumber.FromArray(v);

        /// <summary>NumPy's unit step <c>1.0</c> (the <c>dx</c> list's entry when no spacing is given): a Python float,
        ///     weak under NEP 50.</summary>
        private static readonly object UnitSpacing = PolyNumber.FromPython(PyScalar.Float(1.0));

        /// <summary>
        ///     A scalar spacing as NumPy's <c>dx</c> list keeps it: the value classified by the house NEP 50 map
        ///     (<see cref="PolyNumber.FromObject"/>: a C# number is a Python number, <see cref="Half"/> / <c>char</c> /
        ///     <c>decimal</c> a NumPy scalar, a 0-d NDArray an ndarray), boxed; a string or null is returned as it is.
        /// </summary>
        /// <param name="v">The spacing (np.ndim(v) == 0).</param>
        /// <returns>A boxed <see cref="PolyNumber"/>, or <paramref name="v"/> itself for a str or None.</returns>
        /// <remarks>
        ///     NumPy accepts a str or None here (np.ndim of either is 0) and refuses it only where Python computes with it,
        ///     <c>2. * ax_dx</c> (<see cref="TwiceSpacing"/>), after the other spacings, the edge order and the axis' length
        ///     were checked. PolyNumber has no str or None kind (NumPy would build a str / object array from them), so they
        ///     stay raw until that statement.
        /// </remarks>
        private static object SpacingOf(object v) => v is null or string ? v : PolyNumber.FromObject(v);

        /// <summary>
        ///     NumPy's <c>2. * ax_dx</c>, the uniform interior's divisor: CPython arithmetic for a Python number, scalarmath
        ///     for a NumPy scalar (a float16 step doubles in float16), a ufunc for a 0-d array (its result a NumPy scalar).
        /// </summary>
        /// <param name="axDx">The step as <see cref="SpacingOf"/> left it.</param>
        /// <returns>The product, a Python number or a NumPy scalar.</returns>
        /// <exception cref="TypeError">A None step (CPython's <c>unsupported operand type(s) for *: 'float' and
        ///     'NoneType'</c>) or a str step (<c>can't multiply sequence by non-int of type 'float'</c>, from str's reflected
        ///     multiply, as Python reports it).</exception>
        private static PolyNumber TwiceSpacing(object axDx)
        {
            if (axDx is null)
                throw new TypeError("unsupported operand type(s) for *: 'float' and 'NoneType'");
            if (axDx is string)
                throw new TypeError("can't multiply sequence by non-int of type 'float'");
            return Mul(Py(2.0), (PolyNumber)axDx);
        }

        /// <summary>
        ///     Whether a result computed in <paramref name="loop"/> can be written into a component of
        ///     <paramref name="otype"/> by a ufunc's out= (same_kind): every loop but a complex one into a real component.
        /// </summary>
        /// <param name="loop">The dtype the statement computes in.</param>
        /// <param name="otype">The component's dtype.</param>
        /// <returns>False for a complex result into a real component, which NumPy's setitem stores as its real part
        ///     (<see cref="StoreComponent"/>).</returns>
        private static bool StoresInPlace(NPTypeCode loop, DType otype) =>
            loop != NPTypeCode.Complex || otype.typecode == NPTypeCode.Complex;

        /// <summary>
        ///     NumPy's <c>out[s1] = value</c> into a component view: a setitem, whose cast is unsafe, so a complex value
        ///     keeps its real part (NumPy's ComplexWarning, which NumSharp does not raise).
        /// </summary>
        /// <param name="dst">The component's slice view.</param>
        /// <param name="value">The computed values (broadcast into <paramref name="dst"/>).</param>
        private static void StoreComponent(NDArray dst, NDArray value) => np.copyto(dst, value, casting: "unsafe");

        /// <summary>A Python float (weak under NEP 50): the literals of NumPy's gradient code (<c>2.</c>, <c>-1.5</c>, ...).</summary>
        /// <param name="v">The value.</param>
        /// <returns>The number.</returns>
        private static PolyNumber Py(double v) => PolyNumber.FromPython(PyScalar.Float(v));

        /// <summary>Python's <c>a + b</c> with NumPy's semantics for the operands' kinds (<see cref="PolyNumber.Binary"/>).</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The sum.</returns>
        private static PolyNumber Add(PolyNumber a, PolyNumber b) => PolyNumber.Binary(BinaryOp.Add, a, b);

        /// <summary>Python's <c>a * b</c> with NumPy's semantics for the operands' kinds.</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The product.</returns>
        private static PolyNumber Mul(PolyNumber a, PolyNumber b) => PolyNumber.Binary(BinaryOp.Multiply, a, b);

        /// <summary>Python's <c>a / b</c> (true division) with NumPy's semantics for the operands' kinds.</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The quotient.</returns>
        /// <exception cref="DivideByZeroException">Two Python numbers with a zero divisor (CPython's ZeroDivisionError:
        ///     <c>float division by zero</c>); a NumPy scalar divisor gives inf / nan instead.</exception>
        private static PolyNumber Div(PolyNumber a, PolyNumber b) => PolyNumber.Binary(BinaryOp.Divide, a, b);

        // Assign an edge hyperplane at index `at` along `axis` into `outp`, then dispose the value.
        /// <summary>
        ///     Stores one computed edge into the output — NumPy's <c>out[tuple(slice1)] = ...</c> with <c>slice1[axis]</c> the
        ///     edge's integer index — and releases the computed value at once (one hyperplane of the input, see
        ///     <see cref="Edge1"/>).
        /// </summary>
        /// <param name="outp">The output component being filled.</param>
        /// <param name="work">The (converted) input; unused by the store, kept for the call sites' symmetry.</param>
        /// <param name="axis">The axis the edge lies on.</param>
        /// <param name="N">The rank of <paramref name="outp"/>.</param>
        /// <param name="at">The edge's index along <paramref name="axis"/> (0, or -1 for the last, as NumPy writes it).</param>
        /// <param name="value">The edge value, (N-1)-D (0-d for a 1-D output); disposed here.</param>
        /// <remarks>
        ///     The target is the integer-indexed hyperplane, as NumPy's: the value has the rank NumPy's has, so it stores
        ///     along any axis (a length-1 slice of a later axis, (3, 1), would not take an (N-1)-D (3,) value). The store is
        ///     NumPy's setitem, whose cast is unsafe: a complex value keeps its real part in a real component.
        /// </remarks>
        private static void WriteEdge(NDArray outp, NDArray work, int axis, int N, long at, NDArray value)
        {
            outp[BuildAxisIndex(N, axis, at)] = value;
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

        /// <summary>
        ///     NumPy's <c>tuple(slice1)</c> with an integer at the axis: every axis taken whole except
        ///     <paramref name="axis"/>, which takes element <paramref name="k"/> and drops out of the result.
        /// </summary>
        /// <param name="N">The rank.</param>
        /// <param name="axis">The indexed axis.</param>
        /// <param name="k">The index (negative counts from the end).</param>
        /// <returns>A fresh index list for the indexer.</returns>
        private static Slice[] BuildAxisIndex(int N, int axis, long k)
        {
            var slices = new Slice[N];
            for (int i = 0; i < N; i++)
                slices[i] = i == axis ? Slice.Index(k) : Slice.All;
            return slices;
        }

        // ndim of a spacing argument: 0 for a C# scalar, else the NDArray's ndim.
        /// <summary>
        ///     NumPy's <c>np.ndim(v)</c> for a spacing (<c>np.asarray(v).ndim</c>), which decides whether it is a step (0) or
        ///     coordinates: 0 for a C# scalar, a string or null (None), the rank of an NDArray or a typed C# array, 1 for a
        ///     <see cref="Memory{T}"/> of a dtype, and the nesting depth of a Python sequence (a <c>List&lt;T&gt;</c>, an
        ///     <c>object[]</c>, a tuple: the house map, <see cref="PolySequence.IsSequence"/>).
        /// </summary>
        /// <param name="v">The spacing.</param>
        /// <returns>Its rank.</returns>
        /// <exception cref="ValueError">A ragged sequence (np.asarray's inhomogeneous-shape text, raised where NumPy's
        ///     np.ndim raises it).</exception>
        /// <remarks>
        ///     A sequence was read as a scalar (only a C# array was a coordinate list), so a <c>List&lt;double&gt;</c> of
        ///     coordinates reached the double conversion and its InvalidCastException, and an <c>object[]</c> was measured
        ///     by its C# rank, not by what it holds. A None was refused here (ArgumentNullException); NumPy's
        ///     <c>np.ndim(None)</c> is 0, and its <c>2. * ax_dx</c> raises the TypeError.
        /// </remarks>
        private static int GradSpacingNdim(object v)
        {
            // NumPy's np.ndim(None) is 0: a None step is uniform, and NumPy refuses it only in `2. * ax_dx`.
            if (v is null) return 0;
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

        // The scalar steps and the coordinate differences are read as NumPy values (PolyNumber: SpacingOf, and
        // PolyNumber.ScalarOf of `ax_dx[i]`, which reads an element through its own dtype), never converted to a double:
        // NDArray.GetDouble indexes the float64 lane of the storage whatever the dtype (fixed in 382d0738, whose
        // SpacingElement this replaces).

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
