using System.Numerics;
using NumSharp;

namespace NumSharpOpenBlasDefaultExample;

/// <summary>
///     The matrix products that NumSharp sends to OpenBLAS: np.dot, np.matmul and the functions
///     that are built on them, for float64, float32 and complex128.
/// </summary>
/// <remarks>
///     The code is the same as without the package. Only the library that computes the products
///     changes, so the comment beside each line states the result, which is the same in both cases.
///     Every other element type stays on NumSharp's own kernels: an integer product is exact in
///     every order of summation, so a second implementation cannot improve it.
/// </remarks>
internal static class Products
{
    /// <summary>
    ///     Runs each product once. The method prints nothing: read the outcome beside each line.
    /// </summary>
    /// <remarks>
    ///     One <see cref="NDScope"/> owns every array that the method creates and releases the
    ///     buffers when the method returns.
    /// </remarks>
    public static void Run()
    {
        using var scope = NDScope.Open();

        NDArray v = np.array(new double[] { 1, 2, 3 });
        NDArray w = np.array(new double[] { 4, 5, 6 });
        NDArray m = np.arange(6.0).reshape(2, 3);                        // [[0, 1, 2], [3, 4, 5]]

        // ---- np.dot and np.matmul ------------------------------------------------------------
        NDArray scalar = np.dot(v, w);                                   // 32.0, a 0-d array
        NDArray matrixVector = np.dot(m, v);                             // [8, 26]
        NDArray gram = np.matmul(m, m.T);                                // [[5, 14], [14, 50]]
        NDArray inner = np.inner(m, v);                                  // [8, 26]: the last axis of each operand
        NDArray outer = np.outer(v, w);                                  // (3, 3): [[4, 5, 6], [8, 10, 12], [12, 15, 18]]

        // ---- a stack of matrices is one call --------------------------------------------------
        NDArray stack = np.arange(24.0).reshape(2, 3, 4);                // two 3 x 4 matrices
        NDArray grams = np.matmul(stack, np.swapaxes(stack, 1, 2));      // (2, 3, 3); grams[0, 0] is [14, 38, 62]

        // ---- float32 and complex128 take the same path ----------------------------------------
        NDArray single = np.dot(m.astype(np.float32), v.astype(np.float32));      // float32 [8, 26]
        NDArray z = np.array(new Complex[] { new(1, 2), new(3, -1) });
        NDArray conjugated = np.vdot(z, z);                              // 15 + 0j: vdot conjugates its first operand
        NDArray plain = np.dot(z, z);                                    // 5 - 2j: dot does not

        // ---- the functions that are built on the products -------------------------------------
        NDArray contracted = np.einsum("ij,jk->ik", new[] { m, m.T });   // [[5, 14], [14, 50]], the same product as gram
        NDArray tensor = np.tensordot(m, m.T, 1);                        // [[5, 14], [14, 50]]
        NDArray chain = np.linalg.multi_dot(new[] { m, m.T, m });        // (2, 3): [[42, 61, 80], [150, 214, 278]]
        NDArray squared = np.linalg.matrix_power(gram, 2);               // [[221, 770], [770, 2696]]

        // ---- integers stay on NumSharp's own kernels ------------------------------------------
        NDArray integers = np.dot(np.arange(6).reshape(2, 3), np.arange(3));      // int64 [5, 14]
    }
}
