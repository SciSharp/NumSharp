using System;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        //  np.piecewise — evaluate a piecewise-defined function.
        //
        //  Port of NumPy 2.x numpy.piecewise (numpy/lib/_function_base_impl.py):
        //
        //      x = asanyarray(x)
        //      n2 = len(funclist)
        //      ... promote a scalar / bare-array condlist to a list ...
        //      condlist = asarray(condlist, dtype=bool);  n = len(condlist)
        //      if n == n2 - 1:                       # one extra function = default
        //          condelse = ~np.any(condlist, axis=0, keepdims=True)
        //          condlist = concatenate([condlist, condelse]);  n += 1
        //      elif n != n2:
        //          raise ValueError(...)
        //      y = zeros_like(x)                     # RESULT DTYPE == x's dtype
        //      for cond, func in zip(condlist, funclist):
        //          if not callable(func):
        //              y[cond] = func                # scalar / array assignment
        //          else:
        //              vals = x[cond]
        //              if vals.size > 0:             # the func is NOT called on an empty slice
        //                  y[cond] = func(vals, *args, **kw)
        //      return y
        //
        //  NumSharp keeps NumPy's exact structure — a composition, not a new
        //  kernel — because that is what numpy.piecewise itself is: it has no C
        //  loop of its own, it drives zeros_like + boolean fancy get/set (which in
        //  NumSharp are the SIMD IL-kernel-backed BooleanMask / BooleanMaskSet).
        //  So the perf comes for free from those primitives, and every dtype /
        //  layout they support (all 15 dtypes; C / F / strided / transposed /
        //  reversed / broadcast x) is supported here too.
        //
        //  Three behaviours that differ from the sibling np.select and are easy to
        //  get wrong (all probed against NumPy 2.4.2):
        //    * OUTPUT DTYPE is x's dtype (zeros_like), NOT result_type — so a float
        //      scalar func into an int x truncates (piecewise(int32, [c], [1.9]) -> 1).
        //    * LAST true condition WINS (a forward overwrite loop), the OPPOSITE of
        //      select's first-wins.
        //    * a scalar func is assigned as a weak python scalar: a float truncates
        //      into an integer target, an out-of-range integer raises OverflowError,
        //      a complex into a non-complex target raises TypeError.
        // =====================================================================

        /// <summary>
        ///     Evaluate a piecewise-defined function. Wherever <paramref name="condlist"/>[i] is true,
        ///     the output takes <paramref name="funclist"/>[i] applied to (or, for a scalar entry, equal
        ///     to) <paramref name="x"/>. When several conditions overlap the LAST true one wins
        ///     (a forward overwrite — the opposite of <see cref="select"/>'s first-wins). Positions no
        ///     condition covers keep the value 0, unless one extra function is supplied as the default.
        /// </summary>
        /// <param name="x">
        ///     The input domain. The result has <paramref name="x"/>'s shape AND dtype
        ///     (<see cref="zeros_like"/>), so a float-valued function stored into an integer
        ///     <paramref name="x"/> truncates toward zero, exactly as NumPy's <c>y[cond] = …</c> does.
        /// </param>
        /// <param name="condlist">
        ///     The boolean conditions, one per function (a non-bool array is read by nonzero, matching
        ///     NumPy's <c>asarray(condlist, dtype=bool)</c>). Each condition should have
        ///     <paramref name="x"/>'s shape.
        /// </param>
        /// <param name="funclist">
        ///     One entry per condition, or one extra as the default. Each entry is either a
        ///     <b>callable</b> — a <see cref="Func{NDArray, NDArray}"/> called as <c>f(x[cond])</c>, or a
        ///     <see cref="Func{NDArray, TArgs, NDArray}"/> (<c>Func&lt;NDArray, object[], NDArray&gt;</c>)
        ///     called as <c>f(x[cond], args)</c> — or a <b>constant</b>: a C# scalar (a weak python
        ///     scalar: a float truncates into an integer target, an out-of-range integer raises, a
        ///     complex into a non-complex target raises) or an <see cref="NDArray"/> (broadcast / placed
        ///     into the selected slots, unsafe-cast to <paramref name="x"/>'s dtype). If
        ///     <c>funclist.Length == condlist.Length + 1</c> the extra entry is the default, used where
        ///     every condition is false.
        /// </param>
        /// <param name="args">
        ///     Extra positional arguments forwarded to every <c>Func&lt;NDArray, object[], NDArray&gt;</c>
        ///     entry (NumPy's <c>*args</c>). A plain <c>Func&lt;NDArray, NDArray&gt;</c> entry ignores them.
        /// </param>
        /// <returns>A fresh array of <paramref name="x"/>'s shape and dtype.</returns>
        /// <exception cref="ValueError">
        ///     <c>funclist.Length</c> is neither <c>condlist.Length</c> nor <c>condlist.Length + 1</c>.
        /// </exception>
        /// <exception cref="IndexError"><paramref name="condlist"/> is empty (NumPy accesses condlist[0]).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray piecewise(NDArray x, NDArray[] condlist, object[] funclist, params object[] args)
        {
            // a numpy.matrix operand: NumPy fills zeros_like(x), a matrix of x's shape
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((x, condlist, funclist, args), static s => piecewise(s.x, s.condlist, s.funclist, s.args));
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            // NumPy's numpy.piecewise after the null checks: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.Piecewise(x, condlist, funclist, args);
        }

        /// <summary>
        ///     <see cref="piecewise(NDArray, NDArray[], object[], object[])"/> with a single bare
        ///     condition array instead of a list — NumPy's undocumented promotion. A <b>2-D+</b> array
        ///     is split along axis 0 into one condition per row (a pre-stacked
        ///     <c>[c0, c1]</c>); a <b>1-D</b> array is treated as ONE whole condition when
        ///     <paramref name="x"/> is not 0-d, or split element-by-element when <paramref name="x"/> IS
        ///     0-d (matching NumPy's <c>condlist[0]</c> inspection exactly).
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray piecewise(NDArray x, NDArray condlist, object[] funclist, params object[] args)
        {
            // a numpy.matrix operand: NumPy fills zeros_like(x), a matrix of x's shape
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((x, condlist, funclist, args), static s => piecewise(s.x, s.condlist, s.funclist, s.args));
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            // NumPy's numpy.piecewise after the null checks: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.Piecewise(x, condlist, funclist, args);
        }

        /// <summary>
        ///     <see cref="piecewise(NDArray, NDArray[], object[], object[])"/> with a single scalar
        ///     boolean condition — NumPy's <c>isscalar(condlist)</c> branch. The one function is applied
        ///     to all of <paramref name="x"/> when <paramref name="condlist"/> is true, and nowhere when
        ///     it is false.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray piecewise(NDArray x, bool condlist, object[] funclist, params object[] args)
        {
            // a numpy.matrix operand: NumPy fills zeros_like(x), a matrix of x's shape
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((x, condlist, funclist, args), static s => piecewise(s.x, s.condlist, s.funclist, s.args));
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            // NumPy's numpy.piecewise after the null checks: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.Piecewise(x, condlist, funclist, args);
        }
    }
}
