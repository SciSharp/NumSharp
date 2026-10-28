using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_commonType</c>, the text refusal of the gufunc's input
        ///     casts, its core-dimension checks and its iterator's broadcast of the stacked axes (a 1-D <c>b</c> picks the
        ///     <c>(m,m),(m)-&gt;(m)</c> gufunc, any other rank <c>(m,m),(m,n)-&gt;(m,n)</c>;
        ///     <see cref="GufuncGuard.CheckCoreDimensions"/>), then <see cref="SolveKernel"/> at the loop dtype.
        /// </remarks>
        // Scope: solve casts BOTH operands via ToCommon (two potential dtype-cast temps) before
        // the engine call — [NDScoped] reclaims whichever were minted (a passthrough operand
        // already at the common type is never tracked), yielding the engine's fresh solution.
        [NDScoped]
        public override NDArray Solve(NDArray a, NDArray b)
        {
            AssertStackedSquare(a);

            bool oneDimensional = b.ndim == 1;
            // NumPy's _commonType(a, b) runs before the gufunc, and the gufunc resolves its loop (where its input
            // casts refuse text) before it checks the core dimensions.
            var common = CommonType(a, b);
            ThrowIfTextGufunc(oneDimensional ? "solve1" : "solve", a, b);
            // The gufunc's checks in NumPy's order (PyUFunc_GeneralizedFunctionInternal): each operand's rank, the core
            // sizes, then its iterator's broadcast of the loop axes, whose refusal is the "remapped shapes" text.
            Backends.GufuncGuard.CheckCoreDimensions(oneDimensional ? Backends.GufuncSignature.Solve1 : Backends.GufuncSignature.Solve,
                new[] { a, b, null }, null, null, keepdimsPassed: false, keepdims: false);

            return SolveKernel(ToCommon(a, common), ToCommon(b, common), oneDimensional);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements: the axis move, <c>oldshape = a.shape[-(an - b.ndim):]</c>, the size check, the square system
        ///     <c>a.reshape(prod, prod)</c> against <c>b.ravel()</c> (through the public <c>np.linalg.solve</c>, so the engine's
        ///     <see cref="Solve"/> computes it) and <c>res.shape = oldshape</c>.
        ///     </para>
        ///     <para>
        ///     NumPy takes <c>oldshape</c> with a Python slice whose edges are part of the answer: when <c>a.ndim ==
        ///     b.ndim</c> the slice starts at <c>-0</c>, which is 0, so the whole shape is the unknowns' shape and only an
        ///     operand of 0 or 1 elements passes the size check (<c>tensorsolve(zeros((0, 0)), zeros((0, 0)))</c> is an empty
        ///     (0, 0) solution, <c>tensorsolve([[2.]], [[4.]])</c> is <c>[[2.]]</c>, a 0-d pair a 0-d solution); when <c>b</c>
        ///     has more axes than <c>a</c> the slice starts past them, and the system is <c>(1, 1)</c> against every element of
        ///     <c>b</c>.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public override NDArray Tensorsolve(NDArray a, NDArray b, int[] axes = null)
        {
            // an = a.ndim, read before the axis move (which keeps the rank).
            int an = a.ndim;
            var work = a;
            if (axes is not null)
            {
                // NumPy removes each named axis from the identity permutation and re-appends it,
                // so the requested axes end up rightmost IN THE ORDER GIVEN.
                var order = new System.Collections.Generic.List<int>();
                for (int k = 0; k < a.ndim; k++)
                    order.Add(k);
                foreach (int k in axes)
                {
                    // Python's list.remove raises for a value the list does not hold; List.Remove only returns
                    // false, which used to append a stray axis and surface as transpose's "axes don't match array".
                    if (!order.Remove(k))
                        throw new ValueError("list.remove(x): x not in list");
                    order.Add(k);
                }

                work = np.transpose(a, order.ToArray());
            }

            // oldshape = a.shape[-(an - b.ndim):], Python's slice of the shape tuple: more axes in a than in b keep the
            // last an - b.ndim; as many slice from -0, which is 0, the whole shape; fewer slice from b.ndim - an, past
            // which no dim is left.
            var dims = work.Shape.dimensions;
            int surplus = an - b.ndim;
            int start = surplus > 0 ? b.ndim : surplus == 0 ? 0 : Math.Min(-surplus, an);
            var oldshape = new long[an - start];
            long prod = 1;
            for (int i = start; i < an; i++)
            {
                oldshape[i - start] = dims[i];
                prod *= dims[i];
            }

            // `if a.size != prod ** 2`, compared by division: Python's ints do not overflow, a long's square can.
            long size = work.size;
            if (prod == 0 ? size != 0 : size % prod != 0 || size / prod != prod)
            {
                // Verbatim, run of spaces included: NumPy's literal is a backslash-continued
                // Python string, so the source's indentation lands inside the message.
                throw new LinAlgError(
                    "Input arrays must satisfy the requirement             " +
                    "prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])");
            }

            var square = np.reshape(work, new[] {prod, prod});
            var rhs = np.ravel(b);

            // `res = wrap(solve(a, b)); res.shape = oldshape`: solve's fresh solution relabelled IN PLACE (it keeps owning
            // its data), not a reshaped view of it.
            var res = np.linalg.solve(square, rhs);
            res.shape = oldshape;
            return res;
        }
    }
}
