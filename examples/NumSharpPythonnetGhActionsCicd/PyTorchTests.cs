using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     PyTorch: a tensor over NumSharp's memory, a model trained with autograd on NumSharp data, and tensors
///     read back as NumSharp arrays (a view when PyTorch allows one, a detached copy when it does not).
/// </summary>
[TestClass]
public class PyTorchTests : PythonTest
{
    /// <summary>
    ///     <c>ToTorch()</c> makes a CPU tensor over NumSharp's buffer: an in-place operation in PyTorch
    ///     changes the NumSharp array, and a write in NumSharp is the value PyTorch reads.
    /// </summary>
    [TestMethod]
    public void TensorSharesNumSharpMemory()
    {
        using var scope = NDScope.Open();
        NDArray grid = np.arange(6).astype(np.float64).reshape(2, 3);
        using (Py.GIL())
        {
            py.Exec("import torch");
            py.tensor = grid.ToTorch();                                      // torch.from_numpy(grid.ToNumpy())
            (long rows, long columns) shape = py.tensor.shape;               // torch.Size reads into a C# tuple
            Assert.AreEqual((2L, 3L), shape);

            py.Exec("tensor.mul_(2.0)");                                     // PyTorch doubles every element in place
            Assert.AreEqual(10.0, grid.GetDouble(1, 2), "NumSharp sees the in-place operation");

            grid[0, 0] = -1.0;
            double inTorch = py.Eval("tensor[0, 0].item()");
            Assert.AreEqual(-1.0, inTorch, "PyTorch sees the write from NumSharp");
        }
    }

    /// <summary>
    ///     Gradient descent with autograd fits a linear model to NumSharp data. The learned weights require
    ///     gradients, so PyTorch refuses to share them; <c>ToTorchNDArray(force: true)</c> detaches and copies.
    /// </summary>
    [TestMethod]
    public void AutogradTrainsOnNumSharpData()
    {
        using var scope = NDScope.Open();
        np.random.seed(11);
        NDArray features = np.random.uniform(-1.0, 1.0, new Shape(40, 3));
        NDArray weights = np.array(new[] { 1.5, -2.0, 0.5 });
        NDArray target = np.dot(features, weights) + 4.0;
        using (Py.GIL())
        {
            py.Exec("import torch");
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
            Assert.ThrowsExactly<PythonException>(() => w.AsTorchNDArray(), "a tensor that requires gradients cannot be viewed");
            NDArray learned = w.ToTorchNDArray(force: true);                 // detach().cpu().numpy(), then a NumSharp copy
            double offset = py.Eval("b.item()"), loss = py.Eval("loss.item()");

            for (int i = 0; i < 3; i++)
                Assert.AreEqual(weights.GetDouble(i), learned.GetDouble(i), 1e-6);
            Assert.AreEqual(4.0, offset, 1e-6);
            Assert.IsTrue(loss < 1e-12, $"the loss after 400 steps is {loss:E2}");
        }
    }

    /// <summary>
    ///     A tensor that PyTorch owns reads into an <see cref="NDArray"/> as a view (writes cross in both
    ///     directions), and the result of a PyTorch function over NumSharp data comes back the same way.
    /// </summary>
    [TestMethod]
    public void TensorsComeBackAsArrays()
    {
        using var scope = NDScope.Open();
        NDArray logits = np.array(new double[,] { { 2.0, 1.0, 0.1 }, { 0.5, 0.5, 0.5 } });
        using (Py.GIL())
        {
            py.Exec("import torch");
            py.Exec("owned = torch.arange(12, dtype=torch.float64).reshape(3, 4)");
            NDArray view = py.owned;                                         // the adapter views Tensor.numpy()
            py.Exec("owned[1, 2] = 77.5");
            Assert.AreEqual(77.5, view.GetDouble(1, 2), "the view follows the tensor");
            view[2, 0] = -12.0;
            double inTorch = py.Eval("owned[2, 0].item()");
            Assert.AreEqual(-12.0, inTorch, "and the tensor follows the view");

            NDArray probabilities = py.torch.softmax(logits.ToTorch(), dim: 1);
            NDArray rowSums = np.sum(probabilities, 1);
            Assert.AreEqual(1.0, rowSums.GetDouble(0), 1e-12, "each row of a softmax sums to 1");
            Assert.AreEqual(1.0, rowSums.GetDouble(1), 1e-12);
            Assert.AreEqual(1.0 / 3.0, probabilities.GetDouble(1, 0), 1e-12, "equal logits give equal probabilities");
            Assert.AreEqual(0L, np.argmax(probabilities["0"]), "the largest logit gets the largest probability");
        }
    }
}
