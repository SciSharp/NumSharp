using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     Matplotlib: NumSharp arrays are plotted off-screen (the Agg backend needs no display, so it works on
///     a build server), the figure is saved as a PNG, and the rendered pixels are read as a NumSharp array.
/// </summary>
[TestClass]
public class MatplotlibTests : PythonTest
{
    /// <summary>
    ///     A figure of two NumSharp series is rendered and saved. The canvas of the figure is then viewed as
    ///     a (height, width, 4) uint8 array without a copy, and NumSharp measures how much of it is drawn on.
    ///     The PNG is attached to the test results, which the workflow uploads.
    /// </summary>
    [TestMethod]
    public void FigureIsRenderedOffScreen_AndItsPixelsAreAnArray()
    {
        using var scope = NDScope.Open();
        NDArray time = np.linspace(0.0, 1.0, 200, endpoint: false);
        NDArray measured = np.sin(2 * Math.PI * 3 * time) + 0.4 * np.sin(2 * Math.PI * 40 * time);
        NDArray smoothed = np.convolve(measured, np.ones(new Shape(9)) / 9.0, "same");       // a moving average, computed by NumSharp
        string png = Path.Combine(TestContext.TestRunDirectory!, "numsharp-matplotlib.png");
        using (Py.GIL())
        {
            // Select the backend before pyplot is imported. Agg draws into memory: no window, no display.
            py.Exec("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt");
            py.time = time;
            py.measured = measured;
            py.smoothed = smoothed;
            py.png = png;
            py.Exec("""
                figure, axes = plt.subplots(figsize=(4, 3), dpi=100)
                axes.plot(time, measured, label='measured')
                axes.plot(time, smoothed, label='smoothed')
                axes.legend()
                figure.savefig(png)
                figure.canvas.draw()
                """);

            using PyObject canvas = py.Eval("figure.canvas.buffer_rgba()");                  // a memoryview of the frame buffer
            NDArray pixels = canvas.AsNDArray();                                             // zero-copy
            CollectionAssert.AreEqual(new long[] { 300, 400, 4 }, pixels.shape, "4 x 3 inches at 100 dots per inch, RGBA");
            Assert.AreEqual(NPTypeCode.Byte, pixels.typecode);
            Assert.AreEqual((byte)255, pixels.GetByte(0, 0, 0), "the corner is the white background");

            NDArray ink = np.any(pixels[":, :, :3"] != 255, 2);                              // true where a pixel is not white
            double coverage = np.count_nonzero(ink) / (double)ink.size;
            Assert.IsTrue(coverage > 0.03 && coverage < 0.6, $"axes, two curves and a legend are drawn ({coverage:P1} of the pixels)");

            py.Exec("plt.close(figure)");
        }

        byte[] header = File.ReadAllBytes(png)[..8];
        CollectionAssert.AreEqual(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, header, "the file is a PNG");
        TestContext.AddResultFile(png);
    }

    /// <summary>
    ///     A Matplotlib colormap is a function from numbers to colors: it turns a NumSharp array of levels
    ///     into an (n, 4) array of RGBA values.
    /// </summary>
    [TestMethod]
    public void ColormapTurnsNumbersIntoColors()
    {
        using var scope = NDScope.Open();
        NDArray levels = np.linspace(0.0, 1.0, 5);
        using (Py.GIL())
        {
            py.Exec("import matplotlib");
            NDArray colors = py.matplotlib.colormaps["viridis"](levels);

            CollectionAssert.AreEqual(new long[] { 5, 4 }, colors.shape);
            Assert.AreEqual(0.267004, colors.GetDouble(0, 0), 1e-6, "viridis starts at a dark purple");
            Assert.AreEqual(0.004874, colors.GetDouble(0, 1), 1e-6);
            Assert.AreEqual(0.329415, colors.GetDouble(0, 2), 1e-6);
            Assert.AreEqual(0.993248, colors.GetDouble(4, 0), 1e-6, "and ends at a yellow");
            Assert.AreEqual(1.0, colors.GetDouble(2, 3), "every color is opaque");
        }
    }
}
