using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     Matplotlib: NumSharp arrays are plotted off-screen, the figure is saved as a PNG, and the rendered
///     pixels are read as a NumSharp array without a copy.
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>MatplotlibTests.cs</c> in NumSharpPythonnetGhActionsCicd.
/// </remarks>
static class MatplotlibExample
{
    /// <summary>
    ///     Where <see cref="Run"/> saves the figure: beside the built program, so the file is in a folder
    ///     that source control ignores. Each run replaces it.
    /// </summary>
    public static string FigurePath => Path.Combine(AppContext.BaseDirectory, "numsharp-matplotlib.png");

    /// <summary>Runs the example and writes <see cref="FigurePath"/>.</summary>
    /// <param name="py">A Python namespace in which <c>import numpy as np</c> already ran. It is used only inside <c>using (Py.GIL())</c> blocks.</param>
    /// <exception cref="PythonException">
    ///     A Python statement raised an error, for example because the library is not installed in the
    ///     interpreter that the process embeds.
    /// </exception>
    public static void Run(dynamic py)
    {
        using var scope = NDScope.Open();                                    // every NDArray built below is released when the method returns

        // Select the backend before pyplot is imported. Agg draws into memory: no window opens, and the
        // same code runs on a machine without a display (a build server, a container).
        using (Py.GIL())
            py.Exec("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt");

        // ---- 1. plot NumSharp arrays, save the figure, read its pixels --------------------------------
        NDArray time = np.linspace(0.0, 1.0, 200, endpoint: false);
        NDArray measured = np.sin(2 * Math.PI * 3 * time) + 0.4 * np.sin(2 * Math.PI * 40 * time);
        NDArray smoothed = np.convolve(measured, np.ones(new Shape(9)) / 9.0, "same");   // a moving average, computed by NumSharp
        using (Py.GIL())
        {
            py.time = time;
            py.measured = measured;
            py.smoothed = smoothed;
            py.png = FigurePath;
            py.Exec("""
                figure, axes = plt.subplots(figsize=(4, 3), dpi=100)
                axes.plot(time, measured, label='measured')
                axes.plot(time, smoothed, label='smoothed')
                axes.legend()
                figure.savefig(png)
                figure.canvas.draw()
                """);                                                        // a multi-statement Python program stays Python: it reads like the Matplotlib documentation

            using PyObject canvas = py.Eval("figure.canvas.buffer_rgba()");  // a memoryview of the frame buffer
            NDArray pixels = canvas.AsNDArray();                             // zero-copy: (300, 400, 4) uint8, red-green-blue-alpha for each pixel
            NDArray ink = np.any(pixels[":, :, :3"] != 255, 2);              // (300, 400) bool: true where a pixel is not white
            double coverage = np.count_nonzero(ink) / (double)ink.size;      // about 0.16: the part of the figure that is drawn on (fonts differ a little by platform)
            py.Exec("plt.close(figure)");                                    // Matplotlib keeps every figure alive until it is closed
        }

        // ---- 2. a colormap is a function from numbers to colors ---------------------------------------
        NDArray levels = np.linspace(0.0, 1.0, 5);
        using (Py.GIL())
        {
            NDArray colors = py.matplotlib.colormaps["viridis"](levels);     // (5, 4) float64 RGBA: from a dark purple (0.267, 0.005, 0.329, 1) to a yellow
        }
    }
}
