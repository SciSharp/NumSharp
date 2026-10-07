using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     The view layouts the size-layout replays (<c>FuzzCorpusTests.RandomApiSizeLayouts</c>) hand to every
    ///     <see cref="Shape"/> argument in place of <c>new Shape(dims)</c>: real views' shapes with the SAME dimensions, so the
    ///     size NumPy saw is unchanged while the strides, offset and buffer size are a view's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Why it exists: NumPy's <c>size</c> is a tuple of dimensions, NumSharp's a <see cref="Shape"/> that can also carry a
    ///     view's addressing (<c>rng.integers(0, 100, x.Shape)</c> for <c>x = a[::4]</c>). Samplers that allocated
    ///     <c>np.empty(size)</c> from the Shape as given returned heap memory for a stepped view and transposed draws for an F
    ///     one. The corpus encodes only dimensions, so without these layouts no oracle case could reach that path.
    ///     </para>
    ///     <para>
    ///     The layouts cover each way a view's Shape differs from a fresh one: a step (gaps), an offset, a negative stride, a
    ///     stride-0 broadcast, a buffer larger than the view (a contiguous slice), and the two dense non-C orders (a transpose
    ///     and a 3-D axis swap) that are memory-safe but must still come back C-ordered, because NumPy fills
    ///     <c>np.empty(size)</c> in memory order. A 0-d size uses a synthetic 0-d shape with an offset (NumSharp's integer
    ///     indexing narrows its 0-d results, so no public view produces one today).
    ///     </para>
    /// </remarks>
    internal static class ViewSizeLayouts
    {
        /// <summary>The layouts of a 0-d size.</summary>
        private static readonly string[] Rank0 = { "0-d offset" };

        /// <summary>The layouts of a 1-D size (a 1-D dense layout is both C and F, so there is no transpose).</summary>
        private static readonly string[] Rank1 = { "stepped", "offset stepped", "reversed", "broadcast", "contiguous slice" };

        /// <summary>The layouts of a 2-D size.</summary>
        private static readonly string[] Rank2 = Rank1.Append("transposed").ToArray();

        /// <summary>The layouts of a size of rank 3 or more.</summary>
        private static readonly string[] Rank3 = Rank2.Append("permuted").ToArray();

        /// <summary>The layouts that exist for a size of <paramref name="rank"/> dimensions.</summary>
        /// <param name="rank">The number of dimensions.</param>
        /// <returns>The layout names, in a fixed order.</returns>
        internal static IReadOnlyList<string> Applicable(int rank) => rank switch
        {
            0 => Rank0,
            1 => Rank1,
            2 => Rank2,
            _ => Rank3,
        };

        /// <summary>
        ///     Builds the <paramref name="layout"/> view of <paramref name="dims"/> and returns its shape (the source arrays are
        ///     disposed at once: a <see cref="Shape"/> holds no storage).
        /// </summary>
        /// <param name="dims">The size's dimensions, all non-negative.</param>
        /// <param name="layout">A name from <see cref="Applicable"/> for this rank.</param>
        /// <returns>A view's shape whose dimensions equal <paramref name="dims"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="layout"/> is unknown or needs more dimensions.</exception>
        /// <exception cref="InvalidOperationException">The built view's dimensions differ from <paramref name="dims"/> (a
        ///     harness bug: the replay would then compare a different size against NumPy).</exception>
        internal static Shape Build(long[] dims, string layout)
        {
            int k = dims.Length;
            string Every(string slice) => string.Join(", ", Enumerable.Repeat(slice, k));
            long size = dims.Aggregate(1L, (a, b) => a * b);

            Shape shape;
            switch (layout)
            {
                case "0-d offset" when k == 0:
                    // What a 0-d view at element 3 of a 20-element buffer looks like (internal constructor, no public route).
                    return new Shape(Array.Empty<long>(), Array.Empty<long>(), 3, 20);
                case "stepped" when k >= 1:
                    shape = ViewShape(() => np.zeros(new Shape(dims.Select(d => 2 * d).ToArray())), b => b[Every("::2")]);
                    break;
                case "offset stepped" when k >= 1:
                    shape = ViewShape(() => np.zeros(new Shape(dims.Select(d => 2 * d + 1).ToArray())), b => b[Every("1::2")]);
                    break;
                case "reversed" when k >= 1:
                    shape = ViewShape(() => np.zeros(new Shape(dims)), b => b[Every("::-1")]);
                    break;
                case "broadcast" when k >= 1:
                    shape = ViewShape(() => np.zeros(new Shape(1L)), b => np.broadcast_to(b, new Shape(dims)));
                    break;
                case "contiguous slice" when k >= 1:
                    shape = ViewShape(() => np.zeros(new Shape(size + 3)), b =>
                    {
                        // The slice is an intermediate view: the reshape takes its own reference on the storage.
                        using var slice = b[$":{size}"];
                        return slice.reshape(dims);
                    });
                    break;
                case "transposed" when k >= 2:
                    shape = ViewShape(() => np.zeros(new Shape(Enumerable.Reverse(dims).ToArray())), b => b.T);
                    break;
                case "permuted" when k >= 3:
                {
                    var swapped = (long[])dims.Clone();
                    (swapped[0], swapped[1]) = (swapped[1], swapped[0]);
                    var perm = Enumerable.Range(0, k).ToArray();
                    (perm[0], perm[1]) = (perm[1], perm[0]);
                    shape = ViewShape(() => np.zeros(new Shape(swapped)), b => np.transpose(b, perm));
                    break;
                }
                default:
                    throw new ArgumentException($"layout '{layout}' does not exist for a rank-{k} size", nameof(layout));
            }

            if (!shape.dimensions.SequenceEqual(dims))
                throw new InvalidOperationException(
                    $"layout '{layout}' built dimensions ({string.Join(",", shape.dimensions)}) for a size of ({string.Join(",", dims)})");
            return shape;
        }

        /// <summary>
        ///     The shape of a view of a fresh source array, both disposed before returning: a <see cref="Shape"/> holds no
        ///     storage, so neither array has to outlive the call.
        /// </summary>
        /// <param name="makeSource">Allocates the source array the view is taken of.</param>
        /// <param name="makeView">Builds the view of the source (it must dispose any intermediate view it creates).</param>
        /// <returns>The view's shape.</returns>
        /// <remarks>
        ///     The source is disposed too, not only the view: the array-object oracle's leak sweep
        ///     (UndisposedIntermediateTests) replays every <c>shapearg</c> case with a <c>view:&lt;layout&gt;</c> form, and
        ///     an undisposed source held one pooled buffer out of the pool until the finalizer ran — counted against the
        ///     API under test although the harness leaked it.
        /// </remarks>
        private static Shape ViewShape(Func<NDArray> makeSource, Func<NDArray, NDArray> makeView)
        {
            using var source = makeSource();
            using var view = makeView(source);
            return view.Shape;
        }

        /// <summary>
        ///     The dimensions of the first decoded <see cref="Shape"/> argument of a case (an argument spelled with
        ///     <c>dims</c>), or null when the case passes none or only a negative one (no view has a negative extent, and NumPy
        ///     raises on it before any layout matters).
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>The dimensions, or null.</returns>
        internal static long[] SizeDims(FuzzCorpus.Case c)
        {
            if (!c.Params.TryGetValue("args", out var args))
                return null;
            foreach (var a in args.EnumerateArray())
            {
                if (!a.TryGetProperty("dims", out var d))
                    continue;
                var dims = d.EnumerateArray()
                    .Select(x => long.Parse(x.ValueKind == JsonValueKind.String ? x.GetString() : x.GetRawText(), CultureInfo.InvariantCulture))
                    .ToArray();
                if (dims.All(x => x >= 0))
                    return dims;
            }
            return null;
        }
    }
}
