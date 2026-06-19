#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;

// ADVERSARIAL: broadcast-view reduction fold (DefaultEngine.ReductionOp.cs).
// Rebuild base -> pre-transform -> broadcast_to(target) -> flat reduce -> compare to NumPy.
// "Try to break it": multiple/inner/interleaved broadcast axes, non-contig remainder,
// scalar collapse, all 15 dtypes, NaN/inf/-0, integer prod overflow, keepdims.
string path = args.Length > 0 ? args[0] : "bcast_ref.json";
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
             : s == "-inf" ? double.NegativeInfinity : double.Parse(s, CultureInfo.InvariantCulture);
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
        case "uint64": { var a = new ulong[n];  for (int i=0;i<n;i++) a[i]=arr[i].GetUInt64(); nd=np.array(a); break; }
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

// pre-transform applied BEFORE broadcast_to (creates non-contig base for nc_* cases)
static NDArray Pre(NDArray v, string p) => p switch
{
    "none"   => v,
    "T"      => v.T,
    "rev"    => v["::-1"],
    "revall" => v["::-1, ::-1"],
    "slice"  => v["1:-1, 1:-1"],
    _ => throw new Exception("pre "+p)
};

static decimal ToDec(object o) => o switch
{
    bool b => b ? 1m : 0m,
    byte v => v, sbyte v => v, short v => v, ushort v => v,
    int v => v, uint v => v, long v => v, ulong v => v,
    char v => (int)v, decimal v => v,
    Half v => (decimal)(double)v, float v => (decimal)v, double v => (decimal)v,
    _ => throw new Exception("ToDec "+o?.GetType().Name)
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
    string pre = c.GetProperty("pre").GetString();
    string rkind = c.GetProperty("rkind").GetString();
    bool kd = c.GetProperty("keepdims").GetBoolean();
    var exp = c.GetProperty("expected");
    var expShape = c.GetProperty("expected_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
    int[] baseShape = c.GetProperty("base_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
    int[] target = c.GetProperty("target").EnumerateArray().Select(x=>x.GetInt32()).ToArray();

    NDArray R = null; string err = null;
    try
    {
        var v = Pre(Build(c.GetProperty("base"), dt, baseShape), pre);
        var bc = np.broadcast_to(v, new Shape(target));
        R = (op, kd) switch
        {
            ("sum",  false) => np.sum(bc),
            ("prod", false) => np.prod(bc),
            ("min",  false) => np.amin(bc),
            ("max",  false) => np.amax(bc),
            ("mean", false) => np.mean(bc),
            ("sum",  true)  => np.sum(bc, true),
            ("min",  true)  => np.amin(bc, (int?)null, true),
            ("prod", true)  => np.prod(bc, keepdims: true),
            ("max",  true)  => np.amax(bc, (int?)null, true),
            ("mean", true)  => np.mean(bc, true),
            _ => throw new Exception(op)
        };
    }
    catch (Exception e) { err = e.GetType().Name + ":" + e.Message.Split('\n')[0]; }

    if (err != null) { fail++; fails.Add($"{id} [{dt}] THREW {err}"); continue; }

    NDArray Rc;
    try { Rc = R.copy(); } catch (Exception e) { fail++; fails.Add($"{id} [{dt}] COPY THREW {e.GetType().Name}:{e.Message.Split('\n')[0]}"); continue; }
    long sz = Rc.size;
    int expN = exp.GetArrayLength();
    if (sz != expN) { fail++; fails.Add($"{id} [{dt}] size {sz}!={expN} (shape [{string.Join(",",Rc.shape)}] vs [{string.Join(",",expShape)}])"); continue; }

    var expArr = exp.EnumerateArray().ToArray();
    bool ok = true; string why = null;
    for (long i = 0; i < sz; i++)
    {
        object boxed = Rc.GetAtIndex(i);
        if (rkind == "c")
        {
            var cc = (Complex)boxed; var pe = expArr[i];
            double er = ParseF(pe[0]), ei = ParseF(pe[1]);
            double tol = 1e-7*(1+Math.Abs(er)+Math.Abs(ei));
            if (!((NaNEq(cc.Real,er)||Math.Abs(cc.Real-er)<=tol) && (NaNEq(cc.Imaginary,ei)||Math.Abs(cc.Imaginary-ei)<=tol)))
                { ok=false; why=$"[{i}] ({cc.Real},{cc.Imaginary})!=({er},{ei})"; break; }
        }
        else if (rkind == "f")
        {
            double a = ToD(boxed), b = ParseF(expArr[i]);
            double rel = dt == "half" ? 3e-2 : dt == "single" ? 1e-4 : 1e-7;
            bool eq = NaNEq(a,b) || Math.Abs(a-b) <= rel*(1+Math.Abs(b));
            if (!eq) { ok=false; why=$"[{i}] {a}!={b}"; break; }
        }
        else // i / u / b : EXACT integer compare (handles overflow wrap past 2^53)
        {
            decimal a = ToDec(boxed);
            decimal b = decimal.Parse(expArr[i].GetRawText(), CultureInfo.InvariantCulture);
            if (a != b) { ok=false; why=$"[{i}] {a}!={b} (exact)"; break; }
        }
    }
    if (ok && expShape.Length >= 0)
    {
        var rs = Rc.shape;
        bool sm = rs.Length == expShape.Length;
        if (sm) for (int d = 0; d < rs.Length; d++) if ((int)rs[d] != expShape[d]) { sm=false; break; }
        if (!sm) { ok=false; why=$"shape [{string.Join(",",rs)}]!=[{string.Join(",",expShape)}]"; }
    }
    if (ok) pass++; else { fail++; fails.Add($"{id} [{dt}] {why}"); }
}

Console.WriteLine($"\n=== broadcast-reduce fold parity: {pass} pass, {fail} fail / {pass+fail} ===");
if (fails.Count > 0)
{
    Console.WriteLine("\n--- FAILURES (grouped prefix:kind) ---");
    foreach (var g in fails.GroupBy(f => {
        var head = f.Split(' ')[0];                         // e.g. nc_T:int64:prod
        var fam = head.Split(':')[0];                       // nc_T
        var kind = f.Contains("THREW")?"THREW":f.Contains("COPY THREW")?"COPY":f.Contains("shape")?"SHAPE":f.Contains("size")?"SIZE":"VALUE";
        return fam+":"+kind;
    }).OrderByDescending(g=>g.Count()))
        Console.WriteLine($"  {g.Key}: {g.Count()}");
    Console.WriteLine("\n--- first 80 ---");
    foreach (var f in fails.Take(80)) Console.WriteLine("  " + f);
}
