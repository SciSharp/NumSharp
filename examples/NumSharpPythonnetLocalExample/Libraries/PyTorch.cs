using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     PyTorch: a tensor over NumSharp's memory, a model trained with autograd on NumSharp data, and tensors
///     read back as NumSharp arrays (a view when PyTorch allows one, a detached copy when it does not).
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>PyTorchTests.cs</c> in NumSharpPythonnetGhActionsCicd.
/// </remarks>
static class PyTorchExample
{
    /// <summary>Runs the example.</summary>
    /// <param name="py">A Python namespace in which <c>import numpy as np</c> already ran. It is used only inside <c>using (Py.GIL())</c> blocks.</param>
    /// <exception cref="PythonException">
    ///     A Python statement raised an error, for example because the library is not installed in the
    ///     interpreter that the process embeds.
    /// </exception>
    public static void Run(dynamic py)
    {
        using var scope = NDScope.Open();                                    // every NDArray built below is released when the method returns
        using (Py.GIL())
            py.Exec("import torch");

        // ---- 1. a tensor over NumSharp's memory -------------------------------------------------------
        NDArray grid = np.arange(6).astype(np.float64).reshape(2, 3);
        using (Py.GIL())
        {
            py.tensor = grid.ToTorch();                                      // torch.from_numpy(grid.ToNumpy()): a CPU tensor over NumSharp's buffer
            (long rows, long columns) shape = py.tensor.shape;               // (2, 3): torch.Size reads into a C# tuple
            py.Exec("tensor.mul_(2.0)");                                     // PyTorch doubles every element in place: grid[1, 2] reads 10.0
            grid[0, 0] = -1.0;                                               // tensor[0, 0].item() reads -1.0
        }

        // ---- 2. autograd trains a linear model on NumSharp data ---------------------------------------
        np.random.seed(11);
        NDArray features = np.random.uniform(-1.0, 1.0, new Shape(40, 3));
        NDArray weights = np.array(new[] { 1.5, -2.0, 0.5 });
        NDArray target = np.dot(features, weights) + 4.0;
        using (Py.GIL())
        {
            py.x = features.ToTorch();
            py.y = target.ToTorch();
            py.Exec("""
                w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
                b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
                for _ in range(400):
                    loss = ((x @ w + b - y) ** 2).mean()
                    loss.backward()                       # autograd fills w.grad and b.grad
                    with torch.no_grad():
                        w -= 0.5 * w.grad
                        b -= 0.5 * b.grad
                        w.grad.zero_()
                        b.grad.zero_()
                """);
            using PyObject w = py.w;
            NDArray learned = w.ToTorchNDArray(force: true);                 // [1.5, -2.0, 0.5]. w requires gradients, so PyTorch refuses to share it
                                                                             // (w.AsTorchNDArray() throws PythonException); force detaches and copies
            double offset = py.Eval("b.item()");                             // 4.0
            double loss = py.Eval("loss.item()");                            // below 1e-12
        }

        // ---- 3. tensors come back as arrays -----------------------------------------------------------
        NDArray logits = np.array(new double[,] { { 2.0, 1.0, 0.1 }, { 0.5, 0.5, 0.5 } });
        using (Py.GIL())
        {
            py.Exec("owned = torch.arange(12, dtype=torch.float64).reshape(3, 4)");
            NDArray view = py.owned;                                         // a view of the tensor PyTorch owns (the adapter views Tensor.numpy())
            py.Exec("owned[1, 2] = 77.5");                                   // view[1, 2] reads 77.5
            view[2, 0] = -12.0;                                              // owned[2, 0].item() reads -12.0
            NDArray probabilities = py.torch.softmax(logits.ToTorch(), dim: 1);   // a PyTorch function over NumSharp data: each row sums to 1
            NDArray rowSums = np.sum(probabilities, 1);                      // [1, 1]: NumSharp checks PyTorch's result
        }
    }
}
