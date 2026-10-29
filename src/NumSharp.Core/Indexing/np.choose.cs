using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Construct an array from an index array and a sequence of arrays to choose from. For every
        ///     position <c>I</c> of the (broadcast) result, the output is
        ///     <c>choices[a[I]][I]</c> — i.e. the value of the <c>a[I]</c>-th choice array at that same
        ///     position. <paramref name="a"/> and every choice are first broadcast to a common shape.
        /// </summary>
        /// <param name="a">
        ///     The index array. Converted to <c>int64</c> under the <c>"safe"</c> casting rule (bool and
        ///     the signed/unsigned integers up to <c>uint32</c> are accepted; <c>uint64</c>, float,
        ///     complex and decimal are rejected with a <see cref="TypeError"/>).
        /// </param>
        /// <param name="choices">
        ///     The choice arrays. Each entry is either an <see cref="NDArray"/> (strong dtype) or a boxed
        ///     C# scalar; as in NumPy (NEP50) an <c>int</c>/<c>float</c>/<c>double</c>/<see cref="Complex"/>
        ///     literal is a weak scalar that adopts the other choices' dtype, while <c>bool</c>,
        ///     <c>char</c>, <see cref="Half"/>, <c>decimal</c> and every <see cref="NDArray"/> are strong.
        ///     All choices are broadcast against each other and against <paramref name="a"/>.
        /// </param>
        /// <param name="out">
        ///     Optional destination. Its shape must EQUAL the broadcast result shape (not merely
        ///     broadcast to it); values are cast into it with unsafe casting and the method returns
        ///     <paramref name="out"/> itself. When <c>null</c> (default) a fresh array is allocated with
        ///     the choices' common dtype.
        /// </param>
        /// <param name="mode">
        ///     How indices outside <c>[0, n-1]</c> are treated: <c>"raise"</c> (default — throw),
        ///     <c>"wrap"</c> (modulo with sign correction) or <c>"clip"</c> (values &lt; 0 → 0,
        ///     values &gt; n-1 → n-1). Case-sensitive, matching NumPy's clip-mode parser.
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array (or <paramref name="out"/>) whose dtype is
        ///     <see cref="result_type"/> of the choices (NEP50) and whose shape is the broadcast of the
        ///     choices against <paramref name="a"/>.
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="choices"/> is empty (<c>"0-length sequence."</c>), an index is out of
        ///     range under <c>mode="raise"</c> (<c>"invalid entry in choice array"</c>), or
        ///     <paramref name="mode"/> is not a known clip mode (NumPy's two clip-mode texts).
        /// </exception>
        /// <exception cref="TypeError">
        ///     <paramref name="a"/> cannot be safe-cast to <c>int64</c>, or <paramref name="out"/> has the
        ///     wrong shape (<c>"choose: invalid shape for output array."</c>).
        /// </exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="choices"/> or a choice entry is null.</exception>
        /// <exception cref="IncorrectShapeException">The choices and <paramref name="a"/> do not broadcast together:
        ///     NumPy's multi-iterator text (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch
        ///     is between arg 0 with shape (0, 3) and arg 2 with shape (0,).</c>), the choices numbered first and the
        ///     index array last.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.choose.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the int64 copy of a non-int64 index array, the 0-d arrays of
        ///     scalar choices, the converted copies of collection choices, the dtype casts of the choices, their
        ///     broadcast views and, when <paramref name="out"/> is given, the computed result that is copied into
        ///     it are all released on return, exceptions included. The result is yielded when <paramref name="out"/>
        ///     is null. Before the scope, only the index copy was released by hand: the casts, the scalar arrays and
        ///     the <c>out=</c> temporary were left to the finalizer, and a broadcast view of a cast choice kept that
        ///     cast's buffer out of the pool. The caller's arrays were built before the scope opened, so the scope
        ///     never disposes them.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, object[] choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            return UnmanagedStorage.ChooseCore(a, choices, @out, mode);
        }

        /// <summary>
        ///     <see cref="choose(NDArray,object[],NDArray,string)"/> for a strongly-typed array of
        ///     choices (the common case — every choice is an <see cref="NDArray"/>). The result dtype is
        ///     <see cref="result_type"/> of the choices.
        /// </summary>
        /// <param name="a">The index array (see <see cref="choose(NDArray,object[],NDArray,string)"/>).</param>
        /// <param name="choices">The choice arrays.</param>
        /// <param name="out">Optional destination whose shape must equal the broadcast result shape.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>A fresh C-contiguous array, or <paramref name="out"/> when given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="choices"/> or a choice is null.</exception>
        /// <exception cref="IncorrectShapeException">The choices and <paramref name="a"/> do not broadcast together:
        ///     NumPy's multi-iterator text (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch
        ///     is between arg 0 with shape (0, 3) and arg 2 with shape (0,).</c>), the choices numbered first and the
        ///     index array last.</exception>
        /// <exception cref="ValueError"><paramref name="choices"/> is empty, an index is out of range under
        ///     <c>mode="raise"</c>, or <paramref name="mode"/> is not a known clip mode.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> cannot be safe-cast to int64, or <paramref name="out"/>
        ///     has the wrong shape.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>, with the same reclaimed set as the <see cref="object"/>[] overload.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, NDArray[] choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            var objs = new object[choices.Length];
            for (int i = 0; i < choices.Length; i++) objs[i] = choices[i];
            return UnmanagedStorage.ChooseCore(a, objs, @out, mode);
        }

        /// <summary>
        ///     <see cref="choose(NDArray,object[],NDArray,string)"/> where <paramref name="choices"/> is a
        ///     single array whose OUTERMOST dimension is taken as the sequence (NumPy's "not recommended"
        ///     abuse — <c>choices.shape[0]</c> choice sub-arrays). A 0-d <paramref name="choices"/> is a
        ///     <see cref="TypeError"/> (<c>"iteration over a 0-d array"</c>), exactly as NumPy raises.
        /// </summary>
        /// <param name="a">The index array (see <see cref="choose(NDArray,object[],NDArray,string)"/>).</param>
        /// <param name="choices">An array whose outermost dimension enumerates the choices.</param>
        /// <param name="out">Optional destination whose shape must equal the broadcast result shape.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>A fresh C-contiguous array, or <paramref name="out"/> when given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="choices"/> is null.</exception>
        /// <exception cref="IncorrectShapeException">The choices and <paramref name="a"/> do not broadcast together:
        ///     NumPy's multi-iterator text (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch
        ///     is between arg 0 with shape (0, 3) and arg 2 with shape (0,).</c>), the choices numbered first and the
        ///     index array last.</exception>
        /// <exception cref="TypeError"><paramref name="choices"/> is 0-d, <paramref name="a"/> cannot be safe-cast
        ///     to int64, or <paramref name="out"/> has the wrong shape.</exception>
        /// <exception cref="ValueError"><paramref name="choices"/> has length 0 along its first axis, an index is out
        ///     of range under <c>mode="raise"</c>, or <paramref name="mode"/> is not a known clip mode.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>, with the same reclaimed set as the <see cref="object"/>[] overload,
        ///     plus the views along axis 0 this overload builds (each held a reference on the caller's buffer until
        ///     the finalizer ran).</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, NDArray choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            if (choices.ndim == 0)
                throw new TypeError("iteration over a 0-d array");
            int n = (int)choices.Shape.dimensions[0];
            var objs = new object[n];
            for (int i = 0; i < n; i++)
                objs[i] = choices[i];   // view along axis 0
            return UnmanagedStorage.ChooseCore(a, objs, @out, mode);
        }
    }
}
