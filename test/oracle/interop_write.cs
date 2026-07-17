#:project ../../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property LangVersion=latest

// Writes a spread of .npy/.npz files with NumSharp for verify_npy_interop.py to read back with real
// NumPy. This is the reverse of the committed oracle: gen_npy_oracle.py proves NumSharp READS NumPy
// and reproduces its bytes, this proves NumPy READS NumSharp — the direction byte-equality cannot
// cover for .npz, whose zip framing legitimately differs.
//
// Run via: python test/oracle/verify_npy_interop.py

using System.Globalization;
using System.Numerics;
using System.Text.Json;
using NumSharp;

string outDir = args.Length > 0 ? args[0] : Path.Combine(Path.GetTempPath(), "numsharp_interop");
Directory.CreateDirectory(outDir);

var manifest = new List<Dictionary<string, object>>();

void Npy(string name, NDArray arr, string expect)
{
    np.save(Path.Combine(outDir, name + ".npy"), arr);
    manifest.Add(new Dictionary<string, object>
    {
        ["file"] = name + ".npy", ["kind"] = "npy", ["expect"] = expect,
        ["dtype"] = arr.typecode.ToString(), ["shape"] = arr.shape,
    });
}

void Npz(string name, Dictionary<string, NDArray> arrays, bool compressed, string expect)
{
    string path = Path.Combine(outDir, name + ".npz");
    if (compressed) np.savez_compressed(path, arrays);
    else np.savez(path, arrays);

    manifest.Add(new Dictionary<string, object>
    {
        ["file"] = name + ".npz", ["kind"] = "npz", ["expect"] = expect,
        ["entries"] = arrays.ToDictionary(kv => kv.Key, kv => (object)new Dictionary<string, object>
        {
            ["dtype"] = kv.Value.typecode.ToString(), ["shape"] = kv.Value.shape,
        }),
    });
}

// --- every dtype NumSharp can write ---
Npy("bool", np.array(new[] { true, false, true }), "[True, False, True]");
Npy("uint8", np.array(new byte[] { 0, 1, 255 }), "[0, 1, 255]");
Npy("int8", np.array(new sbyte[] { -128, 0, 127 }), "[-128, 0, 127]");
Npy("int16", np.array(new short[] { -32768, 0, 32767 }), "[-32768, 0, 32767]");
Npy("uint16", np.array(new ushort[] { 0, 1, 65535 }), "[0, 1, 65535]");
Npy("int32", np.array(new[] { -2147483648, 0, 2147483647 }), "[-2147483648, 0, 2147483647]");
Npy("uint32", np.array(new uint[] { 0, 1, 4294967295 }), "[0, 1, 4294967295]");
Npy("int64", np.array(new[] { long.MinValue, 0L, long.MaxValue }), "[-9223372036854775808, 0, 9223372036854775807]");
Npy("uint64", np.array(new[] { 0UL, 1UL, ulong.MaxValue }), "[0, 1, 18446744073709551615]");
Npy("float16", np.array(new[] { (Half)1.5f, (Half)(-2.25f), Half.PositiveInfinity }), "[1.5, -2.25, inf]");
Npy("float32", np.array(new[] { 1.5f, float.NaN, float.NegativeInfinity }), "[1.5, nan, -inf]");
Npy("float64", np.array(new[] { 1.5, -0.0, double.MaxValue }), "[1.5, -0.0, 1.7976931348623157e+308]");
Npy("complex128", np.array(new[] { new Complex(1, 2), new Complex(-3, double.NaN) }), "[1+2j, -3+nanj]");
Npy("char", np.array(new[] { 'a', 'Z', 'é' }), "['a', 'Z', 'é'] as <U1");

// --- shapes ---
Npy("scalar", np.array(42), "0-d, shape ()");
Npy("empty", np.array(Array.Empty<double>()), "shape (0,)");
Npy("2d", np.arange(6).reshape(2, 3), "shape (2,3)");
Npy("3d", np.arange(24).reshape(2, 3, 4), "shape (2,3,4)");
Npy("empty_2d", np.zeros(new Shape(0, 4), NPTypeCode.Int32), "shape (0,4)");

// --- layouts ---
Npy("fortran", np.arange(6).reshape(2, 3).copy('F'), "F-contiguous, fortran_order: True");
Npy("transposed", np.arange(12).reshape(3, 4).T, "transposed view -> fortran_order: True");
Npy("strided", np.arange(12)["::2"], "strided view -> C-order copy");
Npy("sliced_2d", np.arange(12).reshape(3, 4)["1:, ::2"], "offset+strided view -> C-order copy");

// --- large (chunked write path) ---
Npy("large", np.arange(100_000).astype(typeof(double)), "100k float64, chunked");

// --- npz ---
Npz("npz_single", new() { ["arr_0"] = np.arange(6).reshape(2, 3) }, false, "single member");
Npz("npz_multi", new()
{
    ["weights"] = np.arange(12).reshape(3, 4).astype(typeof(float)),
    ["biases"] = np.array(new[] { 0.5, 1.5 }),
    ["flag"] = np.array(true),
    ["text"] = np.array(new[] { 'h', 'i' }),
}, false, "mixed dtypes");
Npz("npz_compressed", new() { ["big"] = np.zeros(new Shape(50_000), NPTypeCode.Double) }, true, "deflated");
Npz("npz_fortran", new() { ["f"] = np.arange(6).reshape(2, 3).copy('F') }, false, "F-order inside npz");

File.WriteAllText(Path.Combine(outDir, "manifest.json"),
    JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

Console.WriteLine($"{manifest.Count} files -> {outDir}");
