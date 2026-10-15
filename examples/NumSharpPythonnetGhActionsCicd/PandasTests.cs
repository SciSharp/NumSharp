using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     pandas: a DataFrame over NumSharp's memory, and pandas results (frames, columns, group means) read
///     back as NumSharp arrays through the adapter that the interop package has built in.
/// </summary>
[TestClass]
public class PandasTests : PythonTest
{
    /// <summary>Six readings of two sensors: a temperature column and a humidity column.</summary>
    /// <returns>A new (6, 2) float64 array that NumSharp owns.</returns>
    private static NDArray Readings() => np.array(new double[,]
    {
        { 20.5, 61.0 }, { 21.0, 59.5 }, { 22.5, 58.0 }, { 24.0, 55.5 }, { 23.5, 57.0 }, { 21.5, 60.0 },
    });

    /// <summary>
    ///     <c>pd.DataFrame(values, copy=False)</c> puts the frame on NumSharp's buffer: a correction that C#
    ///     writes into the array is the value pandas computes with.
    /// </summary>
    [TestMethod]
    public void DataFrameSitsOnNumSharpMemory()
    {
        using var scope = NDScope.Open();
        NDArray readings = Readings();
        using (Py.GIL())
        {
            py.Exec("import pandas as pd");
            py.readings = readings;
            py.Exec("frame = pd.DataFrame(readings, columns=['temperature', 'humidity'], copy=False)");
            bool shared = py.Eval("np.shares_memory(frame.to_numpy(), readings)");
            Assert.IsTrue(shared, "copy=False: the frame is a view of the NumSharp buffer");

            readings[3, 0] = 30.0;                                    // C# corrects one reading
            double warmest = py.Eval("frame.temperature.max()");
            Assert.AreEqual(30.0, warmest, "pandas computes with the corrected value");
        }
    }

    /// <summary>
    ///     What pandas computes comes back as arrays: a Series, a frame with missing values, a correlation
    ///     matrix and a filtered frame. NumSharp then continues with them (<c>np.nanmean</c>).
    /// </summary>
    [TestMethod]
    public void StatisticsComeBackAsArrays()
    {
        using var scope = NDScope.Open();
        NDArray readings = Readings();
        using (Py.GIL())
        {
            py.Exec("import pandas as pd");
            py.readings = readings;
            py.Exec("frame = pd.DataFrame(readings, columns=['temperature', 'humidity'], copy=False)");

            NDArray means = py.frame.mean();                          // a Series of two values
            Assert.AreEqual(2L, means.size);
            Assert.AreEqual(22.0 + 1.0 / 6.0, means.GetDouble(0), 1e-12);
            Assert.AreEqual(58.5, means.GetDouble(1), 1e-12);

            NDArray rolling = py.Eval("frame.rolling(3).mean()");     // a frame; its first two rows are missing
            CollectionAssert.AreEqual(new long[] { 6, 2 }, rolling.shape);
            Assert.IsTrue(double.IsNaN(rolling.GetDouble(0, 0)) && double.IsNaN(rolling.GetDouble(1, 1)), "pandas marks a missing value with NaN");
            Assert.AreEqual(22.5, rolling.GetDouble(3, 0), 1e-12, "(21.0 + 22.5 + 24.0) / 3");
            NDArray typical = np.nanmean(rolling, axis: 0);           // NumSharp ignores the missing rows
            Assert.AreEqual(22.541666666666668, typical.GetDouble(0), 1e-12);
            Assert.AreEqual(57.875, typical.GetDouble(1), 1e-12);

            NDArray correlation = py.frame.corr();
            Assert.AreEqual(1.0, correlation.GetDouble(0, 0), 1e-12);
            Assert.AreEqual(-0.9742446, correlation.GetDouble(0, 1), 1e-6, "warm air is dry air in this data");

            NDArray warm = py.Eval("frame[frame.temperature > 22.0]");
            CollectionAssert.AreEqual(new long[] { 3, 2 }, warm.shape);
            Assert.AreEqual(24.0, warm.GetDouble(1, 0));
        }
    }

    /// <summary>
    ///     A Series that pandas owns is viewed without a copy. pandas 3 exposes shared memory as read-only
    ///     (its Copy-on-Write rule), so the view must be requested with <c>allowReadonly</c>, and it arrives
    ///     as a NumSharp array that is not writeable.
    /// </summary>
    [TestMethod]
    public void SeriesIsViewedWithoutACopy_AndIsReadOnly()
    {
        using var scope = NDScope.Open();
        using (Py.GIL())
        {
            py.Exec("import pandas as pd");
            py.Exec("temperature = pd.Series([20.5, 21.0, 22.5])");
            using PyObject series = py.temperature;

            Assert.ThrowsExactly<InvalidOperationException>(() => series.AsNDArray(), "a writable view of read-only memory is refused");
            NDArray view = series.AsNDArray(allowReadonly: true);
            Assert.IsFalse(view.Shape.IsWriteable);
            Assert.AreEqual(22.5, view.GetDouble(2));

            py.Exec("temperature.iloc[2] = 25.0");                    // pandas writes in place: no other pandas object shares the values
            Assert.AreEqual(25.0, view.GetDouble(2), "the view follows the Series");

            NDArray copy = series.ToNDArray();                        // an independent array that NumSharp owns
            copy[0] = -1.0;
            double inPandas = py.Eval("temperature.iloc[0]");
            Assert.AreEqual(20.5, inPandas, "a copy does not write back");
        }
    }

    /// <summary>
    ///     Two NumSharp arrays become the columns of a frame (one of them a strided view: the first column
    ///     of the readings), pandas groups by the other, and the group keys and the group means come back as
    ///     an int64 array and a float64 array.
    /// </summary>
    [TestMethod]
    public void GroupByOverNumSharpColumns()
    {
        using var scope = NDScope.Open();
        NDArray sensor = np.array(new long[] { 0, 0, 0, 1, 1, 1 });
        NDArray temperature = Readings()[":, 0"];                     // every second value of the buffer: not contiguous
        using (Py.GIL())
        {
            py.Exec("import pandas as pd");
            py.sensor = sensor;
            py.temperature = temperature;
            py.Exec("by_sensor = pd.DataFrame({'sensor': sensor, 'temperature': temperature}).groupby('sensor').temperature.mean()");

            NDArray keys = py.by_sensor.index;
            NDArray means = py.by_sensor;
            Assert.AreEqual(NPTypeCode.Int64, keys.typecode);
            Assert.AreEqual(0L, keys.GetInt64(0));
            Assert.AreEqual(1L, keys.GetInt64(1));
            Assert.AreEqual(64.0 / 3.0, means.GetDouble(0), 1e-12, "(20.5 + 21.0 + 22.5) / 3");
            Assert.AreEqual(23.0, means.GetDouble(1), 1e-12, "(24.0 + 23.5 + 21.5) / 3");
        }
    }
}
