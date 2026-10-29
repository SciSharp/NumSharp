using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of an <see cref="ogrid"/> index expression: an <b>open mesh</b> — a set of N
        ///     arrays, each 1 in every axis but its own. NumPy's <c>ogrid</c> is polymorphic (a single slice
        ///     yields a bare <c>ndarray</c>, several slices yield a <c>tuple</c>), which C# cannot express from
        ///     one indexer, so this value stands in for both:
        ///     <list type="bullet">
        ///     <item>a single-slice result converts implicitly to a bare <see cref="NDArray"/>;</item>
        ///     <item>any result converts implicitly to <see cref="NDArray"/><c>[]</c> and can be
        ///     <c>Deconstruct</c>ed (<c>var (y, x) = np.ogrid["0:3", "0:5"];</c>) or indexed (<c>[k]</c>).</item>
        ///     </list>
        /// </summary>
        /// <remarks>
        ///     This collapses NumPy's bare-array-vs-1-tuple distinction (Python's <c>ogrid[0:5]</c> vs
        ///     <c>ogrid[0:5,]</c>): both spell <c>np.ogrid["0:5"]</c> here, and the single stored array is
        ///     reachable as an <see cref="NDArray"/> or as a length-1 <see cref="NDArray"/><c>[]</c>.
        /// </remarks>
        public readonly struct OGridResult : INDArrayCarrier
        {
            private readonly NDArray[] _arrays;

            internal OGridResult(NDArray[] arrays) => _arrays = arrays ?? Array.Empty<NDArray>();

            /// <summary>Number of arrays in the mesh (one per slice given to <see cref="ogrid"/>).</summary>
            public int Length => _arrays?.Length ?? 0;

            /// <summary>The k-th mesh array (shape 1 in every axis but the k-th).</summary>
            public NDArray this[int index] => (_arrays ?? Array.Empty<NDArray>())[index];

            /// <summary>Returns the mesh arrays as an <see cref="NDArray"/><c>[]</c>.</summary>
            public NDArray[] ToArray() => _arrays ?? Array.Empty<NDArray>();

            /// <summary>
            ///     Exposes the whole open mesh — the natural form for a multi-slice
            ///     <c>np.ogrid[…]</c> (a single-slice result comes back as a length-1 array).
            /// </summary>
            public static implicit operator NDArray[](OGridResult result) => result.ToArray();

            /// <summary>
            ///     Unwraps a SINGLE-slice result to the bare <see cref="NDArray"/> NumPy returns for
            ///     <c>ogrid[0:5]</c>. Throws for a multi-slice result — that outcome is ambiguous, use the
            ///     <see cref="NDArray"/><c>[]</c> conversion, <c>Deconstruct</c> or indexing instead.
            /// </summary>
            public static implicit operator NDArray(OGridResult result)
            {
                var arrays = result.ToArray();
                if (arrays.Length != 1)
                    throw new InvalidOperationException(
                        $"np.ogrid produced {arrays.Length} arrays; only a single-slice result converts to a " +
                        "bare NDArray. Use the NDArray[] conversion, Deconstruct, or an index instead.");
                return arrays[0];
            }

            /// <summary>Deconstructs a two-slice mesh: <c>var (y, x) = np.ogrid["0:3", "0:5"];</c></summary>
            public void Deconstruct(out NDArray item1, out NDArray item2)
            {
                EnsureArity(2);
                item1 = _arrays[0];
                item2 = _arrays[1];
            }

            /// <summary>Deconstructs a three-slice mesh.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3)
            {
                EnsureArity(3);
                item1 = _arrays[0];
                item2 = _arrays[1];
                item3 = _arrays[2];
            }

            /// <summary>Deconstructs a four-slice mesh.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3, out NDArray item4)
            {
                EnsureArity(4);
                item1 = _arrays[0];
                item2 = _arrays[1];
                item3 = _arrays[2];
                item4 = _arrays[3];
            }

            private void EnsureArity(int n)
            {
                int have = Length;
                if (have != n)
                    throw new InvalidOperationException(
                        $"np.ogrid produced {have} arrays; cannot deconstruct into {n}. Use indexing or the " +
                        "NDArray[] conversion.");
            }

            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(_arrays);
        }

        /// <summary>
        ///     An instance which returns an <b>open</b> multi-dimensional "meshgrid" when indexed, so that only
        ///     one dimension of each returned array is greater than 1. The number and dimensionality of the
        ///     outputs equal the number of indexing slices. If the step length is not a complex number, the
        ///     stop is NOT inclusive; a <b>complex</b> step (e.g. <c>"…:5j"</c>) instead specifies the number
        ///     of points, with the stop INCLUSIVE: point <c>i</c> is <c>i * step + start</c> with
        ///     <c>step = (stop - start) / (N - 1)</c>, as NumPy computes it (not <c>linspace</c>, which forces the
        ///     last point onto the stop).
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.ogrid</c> (the <c>sparse=True</c> instance of
        ///     <c>numpy.lib.index_tricks.nd_grid</c>; its <c>__getitem__</c>'s statements). It shares the
        ///     slice-expression grammar of <see cref="r_"/>: C# has no <c>a[1:5:2]</c> literal, so a slice is written
        ///     as a string — <c>np.ogrid["0:5"]</c> ≙ <c>np.ogrid[0:5]</c> — and one string may carry several
        ///     comma-separated slices (<c>np.ogrid["0:3, 0:5"]</c>). <see cref="Slice"/> objects work as
        ///     entries too. Unlike <see cref="r_"/> there are NO leading directives: every entry is a slice. A field
        ///     written as digits is a Python int (exact at any size), anything else a Python float.
        ///     <para>
        ///     <b>Dtype.</b> All the slices of a multi-slice mesh share ONE dtype — int64 when every field of
        ///     every slice is a Python int, else float64 (an imaginary step counts as a float). This mirrors
        ///     NumPy's single <c>result_type(*num_list)</c> over all the slice bounds: e.g.
        ///     <c>np.ogrid["0:2", "0.0:2"]</c> gives two <b>float64</b> arrays, not one int64 and one float64.
        ///     A single-slice result keeps the literal-driven dtype of <see cref="r_"/>'s slice branch.
        ///     </para>
        ///     <para>
        ///     <b>Values.</b> A single slice is <c>arange(start, stop, step)</c>, or for an imaginary step
        ///     <c>arange(0, N) * step + start</c> (a fresh array). In a mesh each line is
        ///     <c>arange(size) * step + start</c> in the mesh dtype, with the size <c>ceil((stop - start) / step)</c>
        ///     in Python's arithmetic: a negative size is an empty axis, an underflowed quotient a size of 0, and a
        ///     float step's points are NOT arange's <c>start + i * delta</c>.
        ///     </para>
        ///     <para>
        ///     <b>Missing stop.</b> A single slice may omit its stop (<c>np.ogrid["5:"]</c> is
        ///     <c>arange(0, 5)</c>, exactly like <see cref="r_"/>), except with an imaginary step of more than one
        ///     point: Python's <see cref="TypeError"/> for <c>None - start</c>, as in NumPy. A multi-slice mesh needs
        ///     every stop but that of an imaginary step of one point: NumPy leaks an <c>AttributeError</c> when it is
        ///     absent, so NumSharp raises a <see cref="ValueError"/> naming the slice, at the same point.
        ///     </para>
        ///     <para>
        ///     <b>Errors.</b> A zero step is NumPy's ZeroDivisionError, <see cref="DivideByZeroException"/> with
        ///     Python's text (<c>division by zero</c> over ints, <c>float division by zero</c> once a float is
        ///     involved); a NaN, infinite or past-int64 size raises NumPy's text (<c>cannot convert float NaN to
        ///     integer</c>, <c>cannot convert float infinity to integer</c>, <c>Maximum allowed size exceeded</c>).
        ///     </para>
        ///     <para>
        ///     <b>Return.</b> The <see cref="OGridResult"/> stands in for NumPy's array-or-tuple: a
        ///     single-slice result converts implicitly to a bare <see cref="NDArray"/>; a multi-slice result
        ///     converts to <see cref="NDArray"/><c>[]</c>, deconstructs, or indexes.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html
        /// </remarks>
        /// <example>
        /// <code>
        /// NDArray x   = np.ogrid["-1:1:5j"];         // array([-1., -0.5, 0., 0.5, 1.])
        /// var (y, xx) = np.ogrid["0:5", "0:5"];      // y.shape (5,1), xx.shape (1,5)
        /// NDArray[] g = np.ogrid["0:3", "0:4", "0:5"];
        /// NDArray sum = np.ogrid["0:5", "0:5"][0] + np.ogrid["0:5", "0:5"][1];  // broadcast add
        /// </code>
        /// </example>
        public sealed class OGridClass
        {
            internal OGridClass() { }

            /// <summary>
            ///     Expands the slice expression into an open mesh. See <see cref="OGridClass"/> for how a
            ///     Python slice literal is spelled in C#.
            /// </summary>
            /// <param name="key">
            ///     Slice-expression strings (a colon-bearing <c>"start:stop[:step]"</c>, or several such
            ///     comma-separated), and/or <see cref="Slice"/> objects.
            /// </param>
            public OGridResult this[params object[] key] => new OGridResult(NdGridLines(key, "ogrid"));
        }

        /// <summary>
        ///     Parses the index expression shared by <see cref="ogrid"/> and <see cref="mgrid"/> into one
        ///     <see cref="AxisConcatenator.SliceSpec"/> per slice — NumPy's
        ///     <c>numpy.lib.index_tricks.nd_grid.__getitem__</c> argument handling. Slices are spelled as
        ///     colon-bearing strings (one may hold several, comma-separated) or <see cref="Slice"/> objects;
        ///     there are no directives.
        /// </summary>
        /// <param name="key">The indexer's key.</param>
        /// <param name="caller">The public function name, for error wording (<c>"ogrid"</c>/<c>"mgrid"</c>).</param>
        /// <returns>One slice per slice of the key, in order (none for the empty key).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
        /// <exception cref="ArgumentException">An entry that is neither a slice string (a string with a colon) nor a
        ///     <see cref="Slice"/> / <see cref="Slice"/>[]; a malformed slice string.</exception>
        private static List<AxisConcatenator.SliceSpec> ParseGridSpecs(object[] key, string caller)
        {
            if (key is null)
                throw new ArgumentNullException(nameof(key));

            var specs = new List<AxisConcatenator.SliceSpec>(key.Length);

            for (int k = 0; k < key.Length; k++)
            {
                object item = key[k];
                switch (item)
                {
                    case null:
                        throw new ArgumentNullException($"key[{k}]",
                            $"{caller} index entries must not be null.");

                    case string s:
                        if (s.IndexOf(':') < 0)
                            throw new ArgumentException(
                                $"{caller} indices must be slice expressions such as \"0:5\" or \"0:1:5j\"; " +
                                $"got \"{s}\".", nameof(key));

                        foreach (var token in s.Split(','))
                        {
                            if (string.IsNullOrWhiteSpace(token))
                                continue;
                            specs.Add(AxisConcatenator.ParseSliceToken(token));
                        }

                        break;

                    case Slice slice:
                        specs.Add(AxisConcatenator.SliceSpec.FromSlice(slice));
                        break;

                    case Slice[] slices:
                        foreach (var one in slices)
                            specs.Add(AxisConcatenator.SliceSpec.FromSlice(one));
                        break;

                    default:
                        throw new ArgumentException(
                            $"{caller} indices must be slice expressions (colon-bearing strings like \"0:5\", " +
                            $"or Slice objects); got {item.GetType().Name}.", nameof(key));
                }
            }

            return specs;
        }

        /// <summary>
        ///     NumPy's <c>nd_grid.__getitem__</c> except branch, which a key of ONE slice takes (<c>len(key)</c> of a bare
        ///     slice is the TypeError that leads there): <c>arange(start, stop, step)</c> for a real step, and for an
        ///     imaginary step <c>arange(0, length, 1, dtype=typ) * step + start</c> with <c>length = int(abs(step))</c> and
        ///     <c>step = (stop - start) / float(length - 1)</c> unless <c>length == 1</c>.
        /// </summary>
        /// <param name="spec">The slice.</param>
        /// <returns>A fresh 1-D array: arange's (int64 over Python ints, else float64), or float64 points for an
        ///     imaginary step — <c>i * step + start</c>, NOT linspace (the last point is not forced onto the stop:
        ///     <c>np.mgrid[0.3:0.9:7j]</c> ends at 0.9000000000000001).</returns>
        /// <exception cref="TypeError">An imaginary step of other than one point without a stop (Python's
        ///     <c>unsupported operand type(s) for -: 'NoneType' and 'int'</c>, or <c>'float'</c> for a float start).</exception>
        /// <exception cref="ValueError">A NaN imaginary step (<c>cannot convert float NaN to integer</c>); a length past
        ///     npy_intp (<c>Maximum allowed size exceeded</c>); arange's own errors; the allocation guard's.</exception>
        /// <exception cref="OverflowException">An infinite imaginary step (<c>cannot convert float infinity to
        ///     integer</c>); a Python int past the float range (<c>int too large to convert to float</c>).</exception>
        /// <exception cref="DivideByZeroException">A zero real step (arange's Python division).</exception>
        private static NDArray GridSingleSlice(in AxisConcatenator.SliceSpec spec)
        {
            if (!spec.ImaginaryStep)
                return spec.Materialize();

            // step_float = abs(step); step = length = int(step_float): NaN and infinity are int()'s errors.
            BigInteger length = PythonInt.From(Math.Abs(spec.Step));
            // A Python int 1 when length == 1: arange(...) * 1 leaves the points as they are.
            double step = 1.0;
            if (!length.IsOne)
            {
                // (key.stop - start) / float(step - 1): Python's None - start is its TypeError.
                if (!spec.HasStop)
                    throw new TypeError(
                        $"unsupported operand type(s) for -: 'NoneType' and '{(spec.StartIsInt ? "int" : "float")}'");
                step = GridPyNumber.TrueDivide(
                    GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)),
                    GridPyNumber.Float(GridPyNumber.IntToFloat(length - 1)));
            }

            // typ = result_type(start, stop, step_float) is float64 (step_float is a Python float); arange(0, length, 1,
            // dtype=typ) refuses a length past npy_intp before allocating anything.
            if (!AxisConcatenator.SliceSpec.FitsInt64(length))
                throw new ValueError("Maximum allowed size exceeded");
            var points = new NDArray(NPTypeCode.Double, (long)length, false);
            // arange * step: the step is a float, nothing to convert; + start: a Python int start converts here, so an
            // int past the float range raises after the allocation, as NumPy's does.
            GridFillFloat(points, step, GridPyNumber.StartOf(spec).AsFloat());
            return points;
        }

        /// <summary>
        ///     The size statement of NumPy's <c>nd_grid.__getitem__</c> for one slice of a key of several:
        ///     <c>int(abs(step))</c> for an imaginary step, else <c>math.ceil((stop - start) / step)</c> in Python's
        ///     arithmetic over the slice's Python numbers.
        /// </summary>
        /// <param name="spec">The slice.</param>
        /// <param name="k">The slice's position in the key, for the missing-stop error.</param>
        /// <param name="caller">The public entry's name (<c>"mgrid"</c> / <c>"ogrid"</c>), for the missing-stop error.</param>
        /// <returns>The size, a Python int: negative for a range that runs against its step, possibly past int64.</returns>
        /// <exception cref="ValueError">A real step without a stop (NumSharp's text; NumPy's <c>None - start</c> TypeError
        ///     sends it to the except branch, whose <c>key.step</c> raises <c>AttributeError: 'tuple' object has no
        ///     attribute 'step'</c>); a NaN quotient or imaginary step (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="DivideByZeroException">A zero step: <c>division by zero</c> over Python ints, <c>float division
        ///     by zero</c> once a float is involved.</exception>
        /// <exception cref="OverflowException">An infinite quotient or imaginary step (<c>cannot convert float infinity to
        ///     integer</c>); a Python int past the float range met by float arithmetic (<c>int too large to convert to
        ///     float</c>) or an int quotient past it (<c>integer division result too large for a float</c>).</exception>
        private static BigInteger GridAxisSize(in AxisConcatenator.SliceSpec spec, int k, string caller)
        {
            // size.append(int(abs(step))): an imaginary step's magnitude is a Python float.
            if (spec.ImaginaryStep)
                return PythonInt.From(Math.Abs(spec.Step));
            if (!spec.HasStop)
                throw GridMissingStop(caller, k);
            // size.append(math.ceil((stop - start) / step)): math.ceil of a float, whose NaN and infinity are errors.
            double quotient = GridPyNumber.TrueDivide(
                GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)), GridPyNumber.StepOf(spec));
            if (double.IsNaN(quotient))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(quotient))
                throw new OverflowException("cannot convert float infinity to integer");
            return new BigInteger(Math.Ceiling(quotient));
        }

        /// <summary>
        ///     The ONE dtype a multi-slice mesh shares — NumPy's <c>result_type(*num_list)</c> over every slice's
        ///     start, stop and step (an imaginary step contributes its magnitude, a Python float; a missing stop is
        ///     None, float64): int64 when every field of every slice is a Python int, else float64.
        /// </summary>
        /// <param name="specs">The slices.</param>
        /// <returns><see cref="NPTypeCode.Int64"/> or <see cref="NPTypeCode.Double"/>.</returns>
        private static NPTypeCode GridMeshDtype(List<AxisConcatenator.SliceSpec> specs)
        {
            bool integral = true;
            foreach (var spec in specs)
                integral &= spec.Integral;

            return integral ? NPTypeCode.Int64 : NPTypeCode.Double;
        }

        /// <summary>
        ///     The rescale statement of NumPy's <c>nd_grid.__getitem__</c> for axis <paramref name="k"/>,
        ///     <c>nn[k] = nn[k] * step + start</c>, over the axis's line: <paramref name="line"/> holds the coordinates
        ///     <c>0 .. size-1</c> (NumPy's <c>arange(size)</c>, or a layer of <c>indices</c>, which holds its coordinate at
        ///     every position), and receives <c>i * step + start</c> in the mesh dtype.
        /// </summary>
        /// <param name="line">A fresh C-contiguous 1-D array of the mesh dtype, the axis's size long; written in place.</param>
        /// <param name="spec">The axis's slice.</param>
        /// <param name="k">The axis, for the missing-stop error.</param>
        /// <param name="typ">The mesh dtype, int64 or float64 (<see cref="GridMeshDtype"/>).</param>
        /// <param name="caller">The public entry's name, for the missing-stop error.</param>
        /// <exception cref="ValueError">An imaginary step of other than one point without a stop (NumSharp's text; NumPy's
        ///     <c>kk.stop - start</c> TypeError ends in its except branch's AttributeError).</exception>
        /// <exception cref="OverflowException">A Python int step or start past int64 in an int64 mesh (NumPy's
        ///     <c>Python int too large to convert to C long</c>, the LP64 text), or past the float range in a float64 mesh
        ///     (<c>int too large to convert to float</c>) — raised before any element, even of an empty axis.</exception>
        private static void GridFillLine(NDArray line, in AxisConcatenator.SliceSpec spec, int k, NPTypeCode typ,
            string caller)
        {
            if (typ == NPTypeCode.Int64)
            {
                // Every slice is ints with a real step (GridMeshDtype). nn[k] * step converts the step to int64 first, then
                // + start the start: NEP 50's Python-int conversion, whose failure does not depend on the axis's size.
                GridFillInt64(line, GridToInt64(spec.IntStep), GridToInt64(spec.IntStart));
                return;
            }

            double step;
            if (spec.ImaginaryStep)
            {
                // step = int(abs(step)); if step != 1: step = (kk.stop - start) / float(step - 1). The size statement
                // already converted abs(step), so int() cannot raise here.
                BigInteger points = PythonInt.From(Math.Abs(spec.Step));
                if (points.IsOne)
                    step = 1.0;
                else
                {
                    if (!spec.HasStop)
                        throw GridMissingStop(caller, k);
                    step = GridPyNumber.TrueDivide(
                        GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)),
                        GridPyNumber.Float(GridPyNumber.IntToFloat(points - 1)));
                }
            }
            else
                // nn[k] * step: a Python int step converts to float64 here (CPython's float(int), raising past the range).
                step = GridPyNumber.StepOf(spec).AsFloat();

            // + start: the start converts after the step, as NumPy's second ufunc call converts it.
            GridFillFloat(line, step, GridPyNumber.StartOf(spec).AsFloat());
        }

        /// <summary>A Python int as the int64 NumPy's NEP 50 conversion gives it in an int64 operation.</summary>
        /// <param name="value">The int.</param>
        /// <returns>The value.</returns>
        /// <exception cref="OverflowException">The int does not fit int64 (NumPy's <c>Python int too large to convert to C
        ///     long</c> on LP64 platforms; Windows NumPy says <c>int too big to convert</c> — NumSharp pins the LP64 text).</exception>
        private static long GridToInt64(BigInteger value)
        {
            if (!AxisConcatenator.SliceSpec.FitsInt64(value))
                throw new OverflowException("Python int too large to convert to C long");
            return (long)value;
        }

        /// <summary>
        ///     Writes <c>i * step + start</c> over an int64 line: NumPy's int64 multiply and add, which wrap.
        /// </summary>
        /// <param name="line">A fresh C-contiguous int64 line; element 0 is at its storage's base address.</param>
        /// <param name="step">The step.</param>
        /// <param name="start">The start.</param>
        [MethodImpl(Optimize)]
        private static unsafe void GridFillInt64(NDArray line, long step, long start)
        {
            long* p = (long*)line.Storage.Address;
            long n = line.size;
            for (long i = 0; i < n; i++)
                p[i] = unchecked(i * step + start);
        }

        /// <summary>
        ///     Writes <c>i * step + start</c> over a float64 line: NumPy's two float64 ufunc passes, each rounded (a
        ///     product, then a sum; never a fused multiply-add — .NET does not contract them), so a <c>-0.0</c> start
        ///     gives <c>+0.0</c> at <c>i = 0</c> as NumPy's <c>0.0 + -0.0</c> does.
        /// </summary>
        /// <param name="line">A fresh C-contiguous float64 line; element 0 is at its storage's base address.</param>
        /// <param name="step">The step.</param>
        /// <param name="start">The start.</param>
        [MethodImpl(Optimize)]
        private static unsafe void GridFillFloat(NDArray line, double step, double start)
        {
            double* p = (double*)line.Storage.Address;
            long n = line.size;
            for (long i = 0; i < n; i++)
                p[i] = (double)i * step + start;
        }

        /// <summary>
        ///     The error NumSharp raises where NumPy's <c>nd_grid.__getitem__</c> meets a missing stop in a key of several
        ///     slices: Python's <c>None - start</c> TypeError sends NumPy to its except branch, whose <c>key.step</c> on a
        ///     tuple raises <c>AttributeError: 'tuple' object has no attribute 'step'</c> (a documented divergence: the
        ///     ValueError says what is missing). Raised at the same point as NumPy's: a real step's in the size loop, an
        ///     imaginary step's (of other than one point) in the rescale loop.
        /// </summary>
        /// <param name="caller">The public entry's name.</param>
        /// <param name="k">The slice's position in the key.</param>
        /// <returns>The exception, for the caller to throw.</returns>
        private static ValueError GridMissingStop(string caller, int k) =>
            new ValueError(
                $"{caller} with more than one slice requires an explicit stop for slice {k}: 'start:stop' or " +
                "'start:stop:Nj' (NumPy raises AttributeError: 'tuple' object has no attribute 'step').");

        /// <summary>
        ///     A slice field as the Python number NumPy's <c>nd_grid</c> computes with: an int (exact, any size) or a float.
        ///     Only the arithmetic nd_grid's statements perform on fields is modelled, with CPython's rules and texts.
        /// </summary>
        private readonly struct GridPyNumber
        {
            /// <summary>Whether the number is a Python int (else a float).</summary>
            private readonly bool _isInt;

            /// <summary>The int's value.</summary>
            private readonly BigInteger _int;

            /// <summary>The float's value.</summary>
            private readonly double _float;

            /// <summary>Builds a number.</summary>
            /// <param name="isInt">Whether it is a Python int.</param>
            /// <param name="i">The int's value.</param>
            /// <param name="f">The float's value.</param>
            private GridPyNumber(bool isInt, BigInteger i, double f)
            {
                _isInt = isInt;
                _int = i;
                _float = f;
            }

            /// <summary>A Python float.</summary>
            /// <param name="f">The value.</param>
            /// <returns>The number.</returns>
            public static GridPyNumber Float(double f) => new GridPyNumber(false, BigInteger.Zero, f);

            /// <summary>The slice's start (0 when omitted).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The start as its Python number.</returns>
            public static GridPyNumber StartOf(in AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StartIsInt, spec.IntStart, spec.Start);

            /// <summary>The slice's stop (callers check <see cref="AxisConcatenator.SliceSpec.HasStop"/> first).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The stop as its Python number.</returns>
            public static GridPyNumber StopOf(in AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StopIsInt, spec.IntStop, spec.Stop);

            /// <summary>The slice's real step (1 when omitted).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The step as its Python number.</returns>
            public static GridPyNumber StepOf(in AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StepIsInt, spec.IntStep, spec.Step);

            /// <summary>
            ///     CPython's <c>float(int)</c> inside float arithmetic (<c>PyLong_AsDouble</c>): the nearest double, ties to
            ///     even.
            /// </summary>
            /// <param name="value">The int.</param>
            /// <returns>The double.</returns>
            /// <exception cref="OverflowException">Past the double range (<c>int too large to convert to float</c>).</exception>
            public static double IntToFloat(BigInteger value)
            {
                double d = PythonInt.ToDouble(value);
                if (double.IsInfinity(d))
                    throw new OverflowException("int too large to convert to float");
                return d;
            }

            /// <summary>The number as a float: an int converted as CPython's float arithmetic converts it.</summary>
            /// <returns>The double.</returns>
            /// <exception cref="OverflowException">An int past the double range.</exception>
            public double AsFloat() => _isInt ? IntToFloat(_int) : _float;

            /// <summary>
            ///     Python's <c>a - b</c>: exact for two ints, else float subtraction with each int converted first (left,
            ///     then right) and the left NaN kept.
            /// </summary>
            /// <param name="a">The minuend.</param>
            /// <param name="b">The subtrahend.</param>
            /// <returns>The difference.</returns>
            /// <exception cref="OverflowException">An int operand past the double range beside a float.</exception>
            public static GridPyNumber Subtract(GridPyNumber a, GridPyNumber b)
            {
                if (a._isInt && b._isInt)
                    return new GridPyNumber(true, a._int - b._int, 0d);
                double left = a.AsFloat();
                double right = b.AsFloat();
                return Float(PyScalar.FloatSub(left, right));
            }

            /// <summary>
            ///     Python's <c>a / b</c>: the correctly rounded quotient of two ints, else float division with each int
            ///     converted first.
            /// </summary>
            /// <param name="a">The dividend.</param>
            /// <param name="b">The divisor.</param>
            /// <returns>The quotient, a Python float.</returns>
            /// <exception cref="DivideByZeroException">A zero divisor: <c>division by zero</c> for two ints, <c>float
            ///     division by zero</c> otherwise.</exception>
            /// <exception cref="OverflowException">An int quotient past the double range (<c>integer division result too
            ///     large for a float</c>), or an int operand past it beside a float.</exception>
            public static double TrueDivide(GridPyNumber a, GridPyNumber b)
            {
                if (a._isInt && b._isInt)
                    return PyScalar.IntTrueDivide(a._int, b._int);
                double left = a.AsFloat();
                double right = b.AsFloat();
                return PyScalar.FloatDiv(left, right);
            }
        }

        /// <summary>
        ///     NumPy's <c>nd_grid.__getitem__</c> with <c>sparse=True</c> (<c>np.ogrid[...]</c>): one slice is the bare 1-D
        ///     array of the except branch; a key of several computes every size, the shared dtype, one
        ///     <c>arange(size, dtype=typ)</c> per axis, then <c>nn[k] = nn[k] * step + start</c> axis by axis, and returns
        ///     each line as an open-mesh view <c>(1, …, size, …, 1)</c>.
        /// </summary>
        /// <param name="key">The indexer key: slices / slice strings.</param>
        /// <param name="caller">The public entry's name, for error texts.</param>
        /// <returns>The lines: a single bare 1-D array, or one open-mesh view per slice (none for the empty key); the caller
        ///     owns them.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
        /// <exception cref="ArgumentException">An entry that is neither a slice string nor a <see cref="Slice"/>.</exception>
        /// <exception cref="TypeError">See <see cref="GridSingleSlice"/>.</exception>
        /// <exception cref="ValueError">A missing stop (<see cref="GridAxisSize"/>, <see cref="GridFillLine"/>); a NaN size;
        ///     a size past npy_intp (<c>Maximum allowed size exceeded</c>); the allocation guard's.</exception>
        /// <exception cref="DivideByZeroException">A zero real step.</exception>
        /// <exception cref="OverflowException">An infinite size; a Python int NumPy cannot convert (see
        ///     <see cref="GridFillLine"/>).</exception>
        // NDScoped: each line is a temp — only its open-mesh RESHAPE VIEW is returned, and the view holds its own
        // reference on the line's buffer. Without the scope every line's wrapper stranded one pooled buffer per slice
        // (measured: 2 for a 2-slice ogrid). The weaver yields every element of the returned array.
        [NDScoped]
        private static NDArray[] NdGridLines(object[] key, string caller)
        {
            var specs = ParseGridSpecs(key, caller);

            int n = specs.Count;
            // A key of one slice takes NumPy's except branch: a bare 1-D array.
            if (n == 1)
                return new[] { GridSingleSlice(specs[0]) };

            // Every size first, in key order: a zero step, a missing (real-step) stop or a NaN size is raised where
            // NumPy's size loop meets it. The empty key has no slice: NumPy's empty tuple.
            var sizes = new BigInteger[n];
            for (int k = 0; k < n; k++)
                sizes[k] = GridAxisSize(specs[k], k, caller);
            NPTypeCode typ = GridMeshDtype(specs);

            // nn = [arange(size, dtype=typ) for ...], in key order: a size past npy_intp is arange's "Maximum allowed size
            // exceeded" (its length computation overflows), a negative size an empty axis.
            var lines = new NDArray[n];
            for (int k = 0; k < n; k++)
            {
                if (!AxisConcatenator.SliceSpec.FitsInt64(sizes[k]))
                    throw new ValueError("Maximum allowed size exceeded");
                lines[k] = new NDArray(typ, Math.Max((long)sizes[k], 0L), false);
            }

            // nn[k] = nn[k] * step + start, axis by axis (the conversions and the missing stops of imaginary steps raise
            // here, after every line exists, as in NumPy); then each line becomes its open-mesh view.
            var @out = new NDArray[n];
            for (int k = 0; k < n; k++)
            {
                GridFillLine(lines[k], specs[k], k, typ, caller);

                long[] shape = new long[n];
                for (int d = 0; d < n; d++)
                    shape[d] = 1L;
                shape[k] = lines[k].size;

                @out[k] = lines[k].reshape(Shape.Adopt(shape));
            }

            return @out;
        }

        /// <summary>
        ///     Returns an open multi-dimensional "meshgrid" when indexed — see <see cref="OGridClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html</remarks>
        public static OGridClass ogrid { get; } = new OGridClass();
    }
}
