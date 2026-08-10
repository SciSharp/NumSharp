using System;
using System.Collections.Generic;
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
        public readonly struct OGridResult
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
        }

        /// <summary>
        ///     An instance which returns an <b>open</b> multi-dimensional "meshgrid" when indexed, so that only
        ///     one dimension of each returned array is greater than 1. The number and dimensionality of the
        ///     outputs equal the number of indexing slices. If the step length is not a complex number, the
        ///     stop is NOT inclusive; a <b>complex</b> step (e.g. <c>"…:5j"</c>) instead specifies the number
        ///     of points, with the stop INCLUSIVE (i.e. <see cref="linspace"/>).
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.ogrid</c> (the <c>sparse=True</c> instance of
        ///     <c>numpy.lib.index_tricks.nd_grid</c>). It shares the slice-expression grammar of
        ///     <see cref="r_"/>: C# has no <c>a[1:5:2]</c> literal, so a slice is written as a string —
        ///     <c>np.ogrid["0:5"]</c> ≙ <c>np.ogrid[0:5]</c> — and one string may carry several
        ///     comma-separated slices (<c>np.ogrid["0:3, 0:5"]</c>). <see cref="Slice"/> objects work as
        ///     entries too. Unlike <see cref="r_"/> there are NO leading directives: every entry is a slice.
        ///     <para>
        ///     <b>Dtype.</b> All the slices of a multi-slice mesh share ONE dtype — int64 when every field of
        ///     every slice is an integer literal, else float64 (a complex step counts as non-integer). This
        ///     mirrors NumPy's single <c>result_type(*num_list)</c> over all the slice bounds: e.g.
        ///     <c>np.ogrid["0:2", "0.0:2"]</c> gives two <b>float64</b> arrays, not one int64 and one float64.
        ///     A single-slice result keeps the literal-driven dtype of <see cref="r_"/>'s slice branch.
        ///     </para>
        ///     <para>
        ///     <b>Missing stop.</b> A single slice may omit its stop (<c>np.ogrid["5:"]</c> is
        ///     <c>arange(0, 5)</c>, exactly like <see cref="r_"/>). A multi-slice mesh may NOT: NumPy needs the
        ///     stop to size each axis and leaks an <c>AttributeError</c> when it is absent, so NumSharp raises a
        ///     clear <see cref="ValueError"/> for <c>np.ogrid["0:3", "5:"]</c>. An imaginary step always
        ///     requires a stop (single or multi), also a <see cref="ValueError"/>.
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
            public OGridResult this[params object[] key] => new OGridResult(Build(key));

            private static NDArray[] Build(object[] key)
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
                                "ogrid index entries must not be null.");

                        case string s:
                            if (s.IndexOf(':') < 0)
                                throw new ArgumentException(
                                    $"ogrid indices must be slice expressions such as \"0:5\" or \"0:1:5j\"; " +
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
                                "ogrid indices must be slice expressions (colon-bearing strings like \"0:5\", " +
                                $"or Slice objects); got {item.GetType().Name}.", nameof(key));
                    }
                }

                int n = specs.Count;
                if (n == 0)
                    return Array.Empty<NDArray>();

                // A single slice returns a bare 1-D array (NumPy's except-branch): arange, or the
                // linspace of an imaginary step. Missing stop is fine here (arange promotes start to
                // stop) and imaginary-without-stop raises, both inside SliceSpec.Materialize.
                if (n == 1)
                    return new[] { specs[0].Materialize() };

                // A multi-slice mesh shares ONE dtype across all slices (NumPy's single result_type over
                // every slice bound): int64 iff every field of every slice is an integer literal.
                bool integral = true;
                for (int k = 0; k < n; k++)
                    integral &= specs[k].Integral;
                NPTypeCode target = integral ? NPTypeCode.Int64 : NPTypeCode.Double;

                var @out = new NDArray[n];
                for (int k = 0; k < n; k++)
                {
                    var spec = specs[k];

                    // NumPy sizes each axis from (stop - start), so a multi-slice mesh cannot omit the
                    // stop (upstream leaks an AttributeError; NumSharp is explicit).
                    if (!spec.HasStop)
                        throw new ValueError(
                            "ogrid with more than one slice requires an explicit stop for each slice: " +
                            "'start:stop' or 'start:stop:Nj'.");

                    NDArray line = spec.Materialize();
                    if (line.typecode != target)
                        line = line.astype(target);

                    // Open mesh: shape is 1 in every axis but this one.
                    long[] shape = new long[n];
                    for (int d = 0; d < n; d++)
                        shape[d] = 1L;
                    shape[k] = line.size;

                    @out[k] = line.reshape(shape);
                }

                return @out;
            }
        }

        /// <summary>
        ///     Returns an open multi-dimensional "meshgrid" when indexed — see <see cref="OGridClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html</remarks>
        public static OGridClass ogrid { get; } = new OGridClass();
    }
}
