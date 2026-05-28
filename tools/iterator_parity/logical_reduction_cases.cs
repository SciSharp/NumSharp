#:project ../../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using System.Text.Json;
using NumSharp;

static object Snapshot(NDArray nd)
{
    if (nd.ndim == 0)
        return (bool)nd;

    var shape = nd.shape;
    if (nd.ndim == 1)
    {
        var values = new bool[shape[0]];
        for (int i = 0; i < shape[0]; i++)
            values[i] = nd.GetBoolean(i);
        return values;
    }

    if (nd.ndim == 2)
    {
        var values = new bool[shape[0]][];
        for (int i = 0; i < shape[0]; i++)
        {
            values[i] = new bool[shape[1]];
            for (int j = 0; j < shape[1]; j++)
                values[i][j] = nd.GetBoolean(i, j);
        }
        return values;
    }

    throw new NotSupportedException("Harness currently supports up to 2D results.");
}

var transposeSource = np.array(new bool[,] { { true, false, true }, { true, true, false } }).T;
var emptyAxis0 = np.zeros(new long[] { 0, 3 }, NPTypeCode.Boolean);
var emptyAxis1 = np.zeros(new long[] { 2, 0 }, NPTypeCode.Boolean);

var cases = new Dictionary<string, object>
{
    ["all_transpose_axis1"] = Snapshot(np.all(transposeSource, axis: 1)),
    ["all_transpose_axis1_keepdims"] = Snapshot(np.all(transposeSource, axis: 1, keepdims: true)),
    ["any_transpose_axis0"] = Snapshot(np.any(transposeSource, axis: 0)),
    ["any_transpose_axis0_keepdims"] = Snapshot(np.any(transposeSource, axis: 0, keepdims: true)),
    ["all_empty_axis0"] = Snapshot(np.all(emptyAxis0, axis: 0)),
    ["any_empty_axis0"] = Snapshot(np.any(emptyAxis0, axis: 0)),
    ["all_empty_axis1"] = Snapshot(np.all(emptyAxis1, axis: 1)),
    ["any_empty_axis1"] = Snapshot(np.any(emptyAxis1, axis: 1)),
};

Console.WriteLine(JsonSerializer.Serialize(cases));
