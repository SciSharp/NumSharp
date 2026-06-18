#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;

string path = args.Length > 0 ? args[0] : "minmax_ref.json";
var doc = JsonDocument.Parse(System.IO.File.ReadAllText(path));
var cases = doc.RootElement.GetProperty("cases");

static double ParseF(JsonElement e)
{
    if (e.ValueKind == JsonValueKind.True) return 1.0;
    if (e.ValueKind == JsonValueKind.False) return 0.0;
    if (e.ValueKind == JsonValueKind.String)
    {
        var s = e.GetString();
        return s == "nan" ? double.NaN : s == "inf" ? double.PositiveInfinity
             : s == "-inf" ? double.NegativeInfinity : double.Parse(s);
    }
    return e.GetDouble();
}

static NDArray Build(JsonElement vals, string dt, int[] shape)
{
    int n = vals.GetArrayLength();
    var arr = vals.EnumerateArray().ToArray();
    NDArray nd;
    switch (dt)
    {
        case "bool":   { var a = new bool[n];   for (int i=0;i<n;i++) a[i]=arr[i].GetBoolean(); nd=np.array(a); break; }
        case "byte":   { var a = new byte[n];   for (int i=0;i<n;i++) a[i]=(byte)arr[i].GetInt32(); nd=np.array(a); break; }
        case "sbyte":  { var a = new sbyte[n];  for (int i=0;i<n;i++) a[i]=(sbyte)arr[i].GetInt32(); nd=np.array(a); break; }
        case "int16":  { var a = new short[n];  for (int i=0;i<n;i++) a[i]=(short)arr[i].GetInt32(); nd=np.array(a); break; }
        case "uint16": { var a = new ushort[n]; for (int i=0;i<n;i++) a[i]=(ushort)arr[i].GetInt32(); nd=np.array(a); break; }
        case "int32":  { var a = new int[n];    for (int i=0;i<n;i++) a[i]=arr[i].GetInt32(); nd=np.array(a); break; }
        case "uint32": { var a = new uint[n];   for (int i=0;i<n;i++) a[i]=(uint)arr[i].GetInt64(); nd=np.array(a); break; }
        case "int64":  { var a = new long[n];   for (int i=0;i<n;i++) a[i]=arr[i].GetInt64(); nd=np.array(a); break; }
        case "uint64": { var a = new ulong[n];  for (int i=0;i<n;i++) a[i]=(ulong)arr[i].GetInt64(); nd=np.array(a); break; }
        case "char":   { var a = new char[n];   for (int i=0;i<n;i++) a[i]=(char)arr[i].GetInt32(); nd=np.array(a); break; }
        case "single": { var a = new float[n];  for (int i=0;i<n;i++) a[i]=(float)ParseF(arr[i]); nd=np.array(a); break; }
        case "double": { var a = new double[n]; for (int i=0;i<n;i++) a[i]=ParseF(arr[i]); nd=np.array(a); break; }
        case "half":   { var a = new double[n]; for (int i=0;i<n;i++) a[i]=ParseF(arr[i]); nd=np.array(a).astype(NPTypeCode.Half); break; }
        case "decimal":{ var a = new decimal[n];for (int i=0;i<n;i++) a[i]=(decimal)ParseF(arr[i]); nd=np.array(a); break; }
        case "complex":{ var a = new Complex[n];for (int i=0;i<n;i++){ var p=arr[i]; a[i]=new Complex(ParseF(p[0]),ParseF(p[1])); } nd=np.array(a); break; }
        default: throw new Exception("dt "+dt);
    }
    return n == 0 ? nd.reshape(shape) : nd.reshape(shape);
}

static NDArray Xform(NDArray v, string tr) => tr switch
{
    "c" => v,
    "f" => v.copy(order: 'F'),
    "t" => v.T,
    "s2" => v[":, ::2"],
    "s2row" => v["::2, :"],
    "rev" => v["::-1, :"],
    "revcol" => v[":, ::-1"],
    _ => throw new Exception("tr "+tr)
};

static double ToD(object o) => o switch
{
    bool b => b ? 1.0 : 0.0,
    byte v => v, sbyte v => v, short v => v, ushort v => v,
    int v => v, uint v => v, long v => v, ulong v => v,
    char v => (int)v, Half v => (double)v, float v => v, double v => v, decimal v => (double)v,
    _ => double.NaN
};
static bool NaNEq(double a, double b) => (double.IsNaN(a) && double.IsNaN(b)) || a == b;

int pass = 0, fail = 0;
var fails = new List<string>();

foreach (var c in cases.EnumerateArray())
{
    string id = c.GetProperty("id").GetString();
    string dt = c.GetProperty("dtype").GetString();
    string kind = c.GetProperty("kind").GetString();
    bool raises = c.GetProperty("raises").GetBoolean();
    var exp = c.GetProperty("expected");
    var expShape = c.GetProperty("expected_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
    NDArray R = null; string err = null;
    try
    {
        if (kind == "reduce")
        {
            var v = Xform(Build(c.GetProperty("base"), dt, c.GetProperty("base_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray()), c.GetProperty("transform").GetString());
            var axEl = c.GetProperty("axis");
            int? ax = axEl.ValueKind == JsonValueKind.Null ? (int?)null : axEl.GetInt32();
            bool kd = c.GetProperty("keepdims").GetBoolean();
            string op = c.GetProperty("op").GetString();
            R = op == "amin" ? np.amin(v, ax, kd) : np.amax(v, ax, kd);
        }
        else
        {
            var A = Build(c.GetProperty("a"), dt, c.GetProperty("a_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray());
            var B = Build(c.GetProperty("b"), dt, c.GetProperty("b_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray());
            string binop = c.GetProperty("binop").GetString();
            R = binop == "maximum" ? np.maximum(A, B) : np.minimum(A, B);
        }
    }
    catch (Exception e) { err = e.GetType().Name + ":" + e.Message.Split('\n')[0]; }

    if (raises)
    {
        if (err != null) { pass++; } else { fail++; fails.Add($"{id} [{dt}] expected RAISE but got result"); }
        continue;
    }
    if (err != null) { fail++; fails.Add($"{id} [{dt}] THREW {err}"); continue; }

    // value compare (C-order)
    var Rc = R.copy();
    long sz = Rc.size;
    int expN = exp.GetArrayLength();
    if (sz != expN) { fail++; fails.Add($"{id} [{dt}] size {sz} != expected {expN} (shape [{string.Join(",",Rc.shape)}] vs [{string.Join(",",expShape)}])"); continue; }
    bool ok = true; string why = null;
    var expArr = exp.EnumerateArray().ToArray();
    for (long i = 0; i < sz; i++)
    {
        object boxed = Rc.GetAtIndex(i);
        if (dt == "complex")
        {
            var cc = (Complex)boxed;
            var pe = expArr[i];
            double er = ParseF(pe[0]), ei = ParseF(pe[1]);
            if (!(NaNEq(cc.Real, er) && NaNEq(cc.Imaginary, ei))) { ok=false; why=$"[{i}] ({cc.Real},{cc.Imaginary})!=({er},{ei})"; break; }
        }
        else
        {
            double a = ToD(boxed), b = ParseF(expArr[i]);
            // tolerance only for half (exact otherwise)
            bool eq = dt == "half" ? (NaNEq(a,b) || Math.Abs(a-b) <= 1e-2*(1+Math.Abs(b))) : NaNEq(a,b);
            if (!eq) { ok=false; why=$"[{i}] {a}!={b}"; break; }
        }
    }
    // shape compare (skip scalar)
    if (ok && expShape.Length > 0)
    {
        var rs = Rc.shape;
        bool sm = rs.Length == expShape.Length;
        if (sm) for (int d = 0; d < rs.Length; d++) if ((int)rs[d] != expShape[d]) { sm = false; break; }
        if (!sm) { ok=false; why=$"shape [{string.Join(",",rs)}]!=[{string.Join(",",expShape)}]"; }
    }
    if (ok) pass++; else { fail++; fails.Add($"{id} [{dt}] {why}"); }
}

Console.WriteLine($"\n=== min/max parity: {pass} pass, {fail} fail / {pass+fail} ===");
if (fails.Count > 0)
{
    Console.WriteLine("\n--- FAILURES (grouped) ---");
    foreach (var g in fails.GroupBy(f => f.Split(' ')[0].Split(':')[0] + ":" + (f.Contains("THREW")?"THREW":f.Contains("shape")?"SHAPE":f.Contains("size")?"SIZE":f.Contains("RAISE")?"RAISE":"VALUE")).OrderByDescending(g=>g.Count()))
        Console.WriteLine($"  {g.Key}: {g.Count()}");
    Console.WriteLine("\n--- first 50 ---");
    foreach (var f in fails.Take(50)) Console.WriteLine("  " + f);
}

