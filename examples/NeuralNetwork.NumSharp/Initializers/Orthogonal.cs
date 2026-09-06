using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Initializers
{
    /// <summary>
    /// Orthogonal initializer (Saxe et al., 2014; Keras semantics). Draws a
    /// Gaussian matrix and orthonormalizes it, so the returned 2-D matrix has
    /// orthonormal rows (if rows &lt; cols) or columns (otherwise). Shapes with
    /// rank &gt; 2 are flattened Keras-style to (prod(shape[:-1]), shape[-1]),
    /// initialized, and reshaped back.
    ///
    /// Keras/NumPy implement this with QR; NumSharp core has no np.linalg.qr
    /// yet (see ROADMAP core backlog), so this uses modified Gram-Schmidt in
    /// float64 — the R diagonal is positive by construction, which is exactly
    /// the sign convention the QR-based implementations enforce with
    /// `q *= sign(diag(r))` to make Q Haar-distributed.
    /// </summary>
    public class Orthogonal : BaseInitializer
    {
        public float Gain { get; }

        public Orthogonal(float gain = 1.0f) : base("orthogonal")
        {
            Gain = gain;
        }

        public override NDArray Initialize(Shape shape)
        {
            if (shape.NDim < 2)
                throw new ArgumentException(
                    "Orthogonal initializer requires at least a 2-D shape (got rank " + shape.NDim + ").",
                    nameof(shape));

            long rowsL = 1;
            for (int i = 0; i < shape.NDim - 1; i++)
                rowsL *= shape[i];
            int rows = checked((int)rowsL);
            int cols = checked((int)shape[shape.NDim - 1]);

            // Orthonormalize the tall orientation, transpose back if needed.
            bool transpose = rows < cols;
            int r = Math.Max(rows, cols);
            int c = Math.Min(rows, cols);

            NDArray q = DrawOrthonormalColumns(r, c);      // (r, c) float64, QᵀQ = I
            if (transpose)
                q = q.transpose().copy();                  // (c, r) = (rows, cols), orthonormal rows

            NDArray result = (q * (double)Gain).astype(NPTypeCode.Single);
            return shape.NDim == 2 ? result : np.reshape(result, shape);
        }

        /// <summary>
        /// (r, c) matrix with orthonormal columns via modified Gram-Schmidt on
        /// a standard-normal draw, computed in float64.
        /// </summary>
        private static unsafe NDArray DrawOrthonormalColumns(int r, int c)
        {
            NDArray a = np.random.normal(0.0, 1.0, new Shape(r, c));  // float64, contiguous
            double* m = (double*)a.Unsafe.Address;

            for (int j = 0; j < c; j++)
            {
                // Subtract projections onto the already-orthonormal columns.
                for (int k = 0; k < j; k++)
                {
                    double dot = 0;
                    for (int i = 0; i < r; i++)
                        dot += m[i * c + k] * m[i * c + j];
                    for (int i = 0; i < r; i++)
                        m[i * c + j] -= dot * m[i * c + k];
                }

                double norm = 0;
                for (int i = 0; i < r; i++)
                    norm += m[i * c + j] * m[i * c + j];
                norm = Math.Sqrt(norm);

                // A vanishing norm means the column was (numerically) linearly
                // dependent — probability ~0 for Gaussian draws; redraw it.
                if (norm < 1e-12)
                {
                    for (int i = 0; i < r; i++)
                        m[i * c + j] = np.random.normal(0.0, 1.0, new Shape(1)).GetDouble(0);
                    j--;
                    continue;
                }

                double inv = 1.0 / norm;
                for (int i = 0; i < r; i++)
                    m[i * c + j] *= inv;
            }

            return a;
        }
    }
}
