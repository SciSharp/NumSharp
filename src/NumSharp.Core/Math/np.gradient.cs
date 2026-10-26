using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp
{
    // ============================== np.gradient ==============================
    // The body is GradientEngine (Math/GradientEngine.cs), an operation class beside this facade.
    //
    // The API mirrors NumPy's `gradient(f, *varargs, axis=None, edge_order=1)`.
    // C# cannot express NumPy's keyword-only `axis`/`edge_order` alongside the
    // positional `*varargs`, so the spacing is spread over arity overloads
    // (0..3 positional + an explicit object[] for N) and `axis`/`edge_order` are
    // named. Two documented C# resolution quirks fall out of this (both because
    // NumPy makes `axis` keyword-only and C# cannot):
    //   * `gradient(f, 2, 3)` — two BARE INT positional spacings — binds
    //     spacing=2, axis=3 (a single spacing with an integer axis). For two
    //     scalar spacings write doubles `gradient(f, 2.0, 3.0)` or an explicit
    //     `gradient(f, new object[]{2, 3})`.
    //   * A tuple `axis` with NO spacing needs the explicit-array form:
    //     `gradient(f, Array.Empty<object>(), axis: new[]{0, 1})` (a scalar axis
    //     `gradient(f, axis: 0)` works directly).
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="gradient(NDArray, object, int?, int)"/>: NumPy returns a
        ///     single ndarray when the gradient is taken along ONE axis and a tuple of ndarrays
        ///     (one per axis) otherwise. C# cannot pick a return type from a runtime count, so this
        ///     value stands in for both — it converts implicitly to <see cref="NDArray"/> (valid only
        ///     for the single-axis result), to <see cref="NDArray"/><c>[]</c> (always),
        ///     <c>Deconstruct</c>s (<c>var (gy, gx) = np.gradient(a);</c>) and indexes (<c>[k]</c>).
        /// </summary>
        public readonly struct GradientResult : INDArrayCarrier
        {
            private readonly NDArray[] _grads;
            private readonly bool _single;

            internal GradientResult(NDArray[] grads, bool single)
            {
                _grads = grads ?? Array.Empty<NDArray>();
                _single = single;
            }

            /// <summary>Number of gradient components (one per requested axis).</summary>
            public int Length => _grads?.Length ?? 0;

            /// <summary>
            ///     True when the gradient was taken along a single axis — NumPy returns a bare
            ///     ndarray then (convertible via the implicit <see cref="NDArray"/> cast), and a
            ///     tuple otherwise.
            /// </summary>
            public bool IsSingle => _single;

            /// <summary>The gradient component along the k-th requested axis.</summary>
            public NDArray this[int index] => (_grads ?? Array.Empty<NDArray>())[index];

            /// <summary>Returns the gradient components as an <see cref="NDArray"/><c>[]</c>.</summary>
            public NDArray[] ToArray() => _grads ?? Array.Empty<NDArray>();

            /// <summary>
            ///     The single-axis gradient. Matches NumPy's bare-ndarray return, valid only when
            ///     exactly one axis was requested; otherwise throws (use the array conversion).
            /// </summary>
            public static implicit operator NDArray(GradientResult result)
            {
                if (!result._single || result.Length != 1)
                    throw new InvalidOperationException(
                        $"np.gradient produced {result.Length} components; it is a tuple, not a single " +
                        "array. Deconstruct it, index it, or convert to NDArray[].");
                return result._grads[0];
            }

            /// <summary>Exposes all gradient components — the tuple NumPy's <c>gradient</c> returns.</summary>
            public static implicit operator NDArray[](GradientResult result) => result.ToArray();

            /// <summary>Deconstructs a two-component result: <c>var (gy, gx) = np.gradient(a);</c></summary>
            public void Deconstruct(out NDArray item1, out NDArray item2)
            {
                EnsureArity(2);
                item1 = _grads[0];
                item2 = _grads[1];
            }

            /// <summary>Deconstructs a three-component result.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3)
            {
                EnsureArity(3);
                item1 = _grads[0];
                item2 = _grads[1];
                item3 = _grads[2];
            }

            private void EnsureArity(int n)
            {
                int have = Length;
                if (have != n)
                    throw new InvalidOperationException(
                        $"np.gradient produced {have} components; cannot deconstruct into {n}. Use " +
                        "indexing or the NDArray[] conversion.");
            }

            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(_grads);
        }

        // ---- Public overloads (spacing spread over arities; axis/edge_order named) ----

        /// <summary>
        ///     Return the gradient of an N-dimensional array using second-order central differences
        ///     in the interior and first/second-order one-sided differences at the boundaries.
        /// </summary>
        /// <param name="f">Input array (samples of a scalar function).</param>
        /// <param name="spacing">
        ///     Optional spacing between samples: a scalar sample distance applied to every requested
        ///     axis, or a 1-D coordinate array giving the sample positions along the (single) requested
        ///     axis. <c>null</c> (the default) means unit spacing.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to take the gradient; <c>null</c> (default) computes it for every
        ///     axis (returning a tuple). Negative axes count from the end.
        /// </param>
        /// <param name="edge_order">Boundary difference order, 1 (default) or 2.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.gradient.html</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="f"/> is <c>null</c>.</exception>
        /// <exception cref="AxisError">An <paramref name="axis"/> is outside <c>[-f.ndim, f.ndim-1]</c>.</exception>
        /// <exception cref="ValueError">
        ///     A repeated <paramref name="axis"/>; <paramref name="edge_order"/> &gt; 2; an axis is shorter
        ///     than <c>edge_order + 1</c>; or a 1-D coordinate spacing whose length or rank does not match
        ///     its axis ("distances must be either scalars or 1d" / "… must match the length …").
        /// </exception>
        /// <exception cref="TypeError">
        ///     The number of positional spacings equals neither 0, 1, nor the number of requested axes
        ///     ("invalid number of arguments").
        /// </exception>
        /// <exception cref="TypeError">
        ///     <paramref name="f"/> is boolean (NumPy's boolean-subtract TypeError from the internal subtract).
        /// </exception>
        public static GradientResult gradient(NDArray f, object spacing = null, int? axis = null, int edge_order = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(f);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return GradientEngine.Gradient(f,
                            spacing is null ? Array.Empty<object>() : new[] { spacing },
                            axis.HasValue ? new[] { axis.Value } : null,
                            edge_order);
        }

        /// <summary>
        ///     Single spacing with a tuple <paramref name="axis"/> — see the main overload. A null
        ///     <paramref name="spacing"/> means no spacing (NumPy's <c>np.gradient(f, axis=(…))</c>), as in the main
        ///     overload; it used to be passed on as a spacing VALUE and refused.
        /// </summary>
        /// <param name="f">The sampled values.</param>
        /// <param name="spacing">One spacing for every axis (a scalar or a coordinate array), or null for unit spacing.</param>
        /// <param name="axis">The axes to differentiate along (negative values count from the end).</param>
        /// <param name="edge_order">The boundary order, 1 or 2.</param>
        /// <returns>One gradient per axis (a single array when the tuple has one axis).</returns>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing, int[] axis, int edge_order = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(f);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return GradientEngine.Gradient(f, spacing is null ? Array.Empty<object>() : new[] { spacing }, axis, edge_order);
        }

        /// <summary>Two per-axis spacings — see the main overload.</summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing0, object spacing1, int[] axis = null, int edge_order = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(f);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return GradientEngine.Gradient(f, new[] { spacing0, spacing1 }, axis, edge_order);
        }

        /// <summary>Three per-axis spacings — see the main overload.</summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing0, object spacing1, object spacing2, int[] axis = null, int edge_order = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(f);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return GradientEngine.Gradient(f, new[] { spacing0, spacing1, spacing2 }, axis, edge_order);
        }

        /// <summary>
        ///     N per-axis spacings passed explicitly (each a scalar or a 1-D coordinate array). Also the
        ///     form for a tuple <paramref name="axis"/> with no spacing: pass an empty array.
        /// </summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object[] varargs, int[] axis = null, int edge_order = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(f);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return GradientEngine.Gradient(f, varargs ?? Array.Empty<object>(), axis, edge_order);
        }
    }
}
