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

// sum/prod/mean parity over offset/strided/negative-stride layouts (root-fix coverage)
string path = args.Length > 0 ? args[0] : "reduce_ref.json";
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
    return nd.reshape(shape);
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
    "slice" => v["1:3, 1:3"],
    "slicestep" => v["1:4:2, 1:5:2"],
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
    string op = c.GetProperty("op").GetString();
    var exp = c.GetProperty("expected");
    var expShape = c.GetProperty("expected_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
    NDArray R = null; string err = null;
    try
    {
        var v = Xform(Build(c.GetProperty("base"), dt, c.GetProperty("base_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray()), c.GetProperty("transform").GetString());
        var axEl = c.GetProperty("axis");
        int? ax = axEl.ValueKind == JsonValueKind.Null ? (int?)null : axEl.GetInt32();
        R = op switch {
            "sum"  => ax.HasValue ? np.sum(v, ax.Value)  : np.sum(v),
            "prod" => ax.HasValue ? np.prod(v, ax.Value) : np.prod(v),
            "mean" => ax.HasValue ? np.mean(v, ax.Value) : np.mean(v),
            _ => throw new Exception(op) };
    }
    catch (Exception e) { err = e.GetType().Name + ":" + e.Message.Split('\n')[0]; }

    if (err != null) { fail++; fails.Add($"{id} [{dt}] THREW {err}"); continue; }

    var Rc = R.copy();
    long sz = Rc.size;
    int expN = exp.GetArrayLength();
    if (sz != expN) { fail++; fails.Add($"{id} [{dt}] size {sz}!={expN} (shape [{string.Join(",",Rc.shape)}] vs [{string.Join(",",expShape)}])"); continue; }
    var expArr = exp.EnumerateArray().ToArray();
    // tolerance: exact for integer sum/prod; relative for floats
    bool floaty = dt is "single" or "double" or "half" or "decimal" or "complex";
    double rel = dt == "half" ? 3e-2 : (dt is "single") ? 1e-4 : 1e-7;
    bool ok = true; string why = null;
    for (long i = 0; i < sz; i++)
    {
        object boxed = Rc.GetAtIndex(i);
        if (dt == "complex")
        {
            var cc = (Complex)boxed; var pe = expArr[i];
            double er = ParseF(pe[0]), ei = ParseF(pe[1]);
            double tol = 1e-7*(1+Math.Abs(er)+Math.Abs(ei));
            if (!((NaNEq(cc.Real,er)||Math.Abs(cc.Real-er)<=tol) && (NaNEq(cc.Imaginary,ei)||Math.Abs(cc.Imaginary-ei)<=tol)))
                { ok=false; why=$"[{i}] ({cc.Real},{cc.Imaginary})!=({er},{ei})"; break; }
        }
        else
        {
            double a = ToD(boxed), b = ParseF(expArr[i]);
            bool eq;
            if (op != "mean" && !floaty) eq = NaNEq(a,b);            // integer sum/prod: exact
            else eq = NaNEq(a,b) || Math.Abs(a-b) <= rel*(1+Math.Abs(b));
            if (!eq) { ok=false; why=$"[{i}] {a}!={b}"; break; }
        }
    }
    if (ok && expShape.Length > 0)
    {
        var rs = Rc.shape;
        bool sm = rs.Length == expShape.Length;
        if (sm) for (int d = 0; d < rs.Length; d++) if ((int)rs[d] != expShape[d]) { sm=false; break; }
        if (!sm) { ok=false; why=$"shape [{string.Join(",",rs)}]!=[{string.Join(",",expShape)}]"; }
    }
    if (ok) pass++; else { fail++; fails.Add($"{id} [{dt}] {why}"); }
}

Console.WriteLine($"\n=== sum/prod/mean parity: {pass} pass, {fail} fail / {pass+fail} ===");
if (fails.Count > 0)
{
    Console.WriteLine("\n--- FAILURES (grouped op:dtype:kind) ---");
    foreach (var g in fails.GroupBy(f => { var p=f.Split(' ')[0].Split(':'); return p[0]+":"+(f.Contains("THREW")?"THREW":f.Contains("shape")?"SHAPE":f.Contains("size")?"SIZE":"VALUE"); }).OrderByDescending(g=>g.Count()))
        Console.WriteLine($"  {g.Key}: {g.Count()}");
    Console.WriteLine("\n--- first 60 ---");
    foreach (var f in fails.Take(60)) Console.WriteLine("  " + f);
}
