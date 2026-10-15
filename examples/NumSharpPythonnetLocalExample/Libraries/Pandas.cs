using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     pandas: a DataFrame over NumSharp's memory, and pandas results (frames, columns, group means) read
///     back as NumSharp arrays through the adapter that the interop package has built in.
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>PandasTests.cs</c> in NumSharpPythonnetGhActionsCicd.
/// </remarks>
static class PandasExample
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
            py.Exec("import pandas as pd");

        // ---- 1. a DataFrame that sits on NumSharp's memory --------------------------------------------
        NDArray readings = np.array(new double[,]                            // six readings of two sensors: temperature, humidity
        {
            { 20.5, 61.0 }, { 21.0, 59.5 }, { 22.5, 58.0 }, { 24.0, 55.5 }, { 23.5, 57.0 }, { 21.5, 60.0 },
        });
        using (Py.GIL())
        {
            py.readings = readings;
            py.Exec("frame = pd.DataFrame(readings, columns=['temperature', 'humidity'], copy=False)");   // copy=False: the frame is a view of the NumSharp buffer
            readings[3, 0] = 30.0;                                           // C# corrects one reading...
            double warmest = py.Eval("frame.temperature.max()");             // 30.0: ...and pandas computes with it
            readings[3, 0] = 24.0;

            // ---- 2. what pandas computes comes back as arrays ------------------------------------------
            NDArray means = py.frame.mean();                                 // a Series: [22.1667, 58.5]
            NDArray rolling = py.Eval("frame.rolling(3).mean()");            // a frame: (6, 2); the first two rows are NaN, pandas' mark for "missing"
            NDArray typical = np.nanmean(rolling, axis: 0);                  // NumSharp continues and ignores the missing rows: [22.5417, 57.875]
            NDArray correlation = py.frame.corr();                           // (2, 2); correlation[0, 1] == -0.974: warm air is dry air in this data
            NDArray warm = py.Eval("frame[frame.temperature > 22.0]");       // the three rows above 22 degrees: (3, 2)
        }

        // ---- 3. a Series that pandas owns: viewed without a copy, and read-only ------------------------
        using (Py.GIL())
        {
            py.Exec("temperature = pd.Series([20.5, 21.0, 22.5])");
            using PyObject series = py.temperature;
            NDArray view = series.AsNDArray(allowReadonly: true);            // zero-copy and not writeable. pandas 3 exposes shared memory read-only (Copy-on-Write),
                                                                             // so AsNDArray() without allowReadonly throws InvalidOperationException
            py.Exec("temperature.iloc[2] = 25.0");                           // view[2] reads 25.0: pandas wrote in place, no other pandas object shares the values
            NDArray copy = series.ToNDArray();                               // an independent, writeable array: a write to it does not reach pandas
        }

        // ---- 4. group-by over two NumSharp columns -----------------------------------------------------
        NDArray sensor = np.array(new long[] { 0, 0, 0, 1, 1, 1 });
        NDArray temperature = readings[":, 0"];                              // the first column of the readings: a strided view, every second value of the buffer
        using (Py.GIL())
        {
            py.sensor = sensor;
            py.temperature = temperature;
            py.Exec("by_sensor = pd.DataFrame({'sensor': sensor, 'temperature': temperature}).groupby('sensor').temperature.mean()");
            NDArray keys = py.by_sensor.index;                               // int64 [0, 1]
            NDArray groupMeans = py.by_sensor;                               // [21.3333, 23.0]
        }
    }
}
