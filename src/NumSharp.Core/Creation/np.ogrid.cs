using System;
using NumSharp.Backends;

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
        ///     <c>numpy.lib.index_tricks.nd_grid</c>; its <c>__getitem__</c>'s statements are
        ///     <see cref="NDArray.ogrid"/>). It shares the
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
            public OGridResult this[params object[] key] => new OGridResult(NDArray.ogrid(key, "ogrid"));
        }

        /// <summary>
        ///     Returns an open multi-dimensional "meshgrid" when indexed — see <see cref="OGridClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html</remarks>
        public static OGridClass ogrid { get; } = new OGridClass();
    }
}
