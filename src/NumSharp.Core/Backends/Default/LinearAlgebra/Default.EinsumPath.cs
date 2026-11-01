// NumPy's np.einsum_path (numpy/_core/einsumfunc.py) as DefaultEngine's member: the operands' shapes, the optimize=
// resolution NumPy runs (its path_type decision order) and the planner's search and report (EinsumPathPlanner, a
// route-for-route port of _parse_einsum_input, _greedy_path / _optimal_path, _flop_count and the info string). The facades
// are LinearAlgebra/np.einsum_path.cs (the string-first, the explicit-optimize and the variadic / sublist spellings).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override (EinsumPath path, string repr) EinsumPath(string subscripts, NDArray[] operands, object optimize)
        {
            if (subscripts is null)
                throw new ArgumentNullException(nameof(subscripts));

            // NumPy plans from the operands' shapes alone: a null list reads as no operand, and the planner's parser then
            // raises NumPy's operand-count text for the subscripts' terms.
            operands ??= Array.Empty<NDArray>();
            long[][] shapes = EinsumPathShapes(operands);
            // optimize= is resolved before the subscripts are parsed, NumPy's order (einsum_path reads path_type first), so a
            // path NumPy does not understand wins over a bad subscripts string.
            EinsumPathPlanner.Directive directive = EinsumPathPlanner.ResolveOptimize(optimize);
            var (path, repr) = EinsumPathPlanner.Compute(subscripts, shapes, directive);
            return (new EinsumPath(path), repr);
        }

        /// <summary>
        ///     The dimensions of each operand, the only thing np.einsum_path reads from them; the arrays are shared, not
        ///     copied (the planner never writes them).
        /// </summary>
        /// <param name="operands">The operands, none of them null.</param>
        /// <returns>One dimension array per operand.</returns>
        /// <exception cref="ValueError">An operand is null (<c>einsum_path operand {i} is null</c>).</exception>
        private static long[][] EinsumPathShapes(NDArray[] operands)
        {
            var shapes = new long[operands.Length][];
            for (int i = 0; i < operands.Length; i++)
            {
                if (operands[i] is null)
                    throw new ValueError($"einsum_path operand {i} is null");
                shapes[i] = operands[i].Shape.dimensions;
            }

            return shapes;
        }
    }
}
