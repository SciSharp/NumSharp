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

// ADVERSARIAL #2: every reduction FAMILY over broadcast views, incl AXIS reductions
// (a separate path from the flat fold), argmax/argmin (fold-EXCLUDED), var/std, all/any,
// count_nonzero, nan-aware, ptp/median. "Try to break it" beyond the flat fold.
string path = args.Length > 0 ? args[0] : "bcast_ax_ref.json";
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

static NDArray Pre(NDArray v, string p) => p switch
{
    "none"  => v,
    "T"     => v.T,
    "rev"   => v.ndim==2 ? v["::-1, ::-1"] : v["::-1, ::-1, ::-1"],
    "slice" => v.ndim==2 ? v["1:-1, 1:-1"] : v["1:-1, 1:-1, 1:-1"],
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
    int axisRaw = c.GetProperty("axis").GetInt32();
    bool flat = axisRaw == -999;
    int ax = axisRaw;
    var exp = c.GetProperty("expected");
    var expShape = c.GetProperty("expected_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
    int[] baseShape = c.GetProperty("base_shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
    int[] target = c.GetProperty("target").EnumerateArray().Select(x=>x.GetInt32()).ToArray();

    // collected result -> boxed values (C-order) + shape
    List<object> got = new(); long[] gotShape = null; string err = null;
    try
    {
        var v = Pre(Build(c.GetProperty("base"), dt, baseShape), pre);
        var bc = np.broadcast_to(v, new Shape(target));
        if (flat)
        {
            switch (op)
            {
                case "sum":  { var r = kd? np.sum(bc,true): np.sum(bc);   got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "prod": { var r = kd? np.prod(bc,keepdims:true): np.prod(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "min":  { var r = kd? np.amin(bc,(int?)null,true): np.amin(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "max":  { var r = kd? np.amax(bc,(int?)null,true): np.amax(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "mean": { var r = kd? np.mean(bc,true): np.mean(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "var":  { var r = np.var(bc,kd); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "std":  { var r = np.std(bc,kd); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "argmax": { got.Add(np.argmax(bc)); gotShape=new long[0]; break; }
                case "argmin": { got.Add(np.argmin(bc)); gotShape=new long[0]; break; }
                case "all":  { got.Add(np.all(bc)); gotShape=new long[0]; break; }
                case "any":  { got.Add(np.any(bc)); gotShape=new long[0]; break; }
                case "count_nonzero": { got.Add(np.count_nonzero(bc)); gotShape=new long[0]; break; }
                case "nansum":  { var r=np.nansum(bc);  got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "nanmax":  { var r=np.nanmax(bc);  got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "nanmin":  { var r=np.nanmin(bc);  got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "nanmean": { var r=np.nanmean(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "ptp":    { var r=np.ptp(bc);    got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                case "median": { var r=np.median(bc); got.Add(r.GetAtIndex(0)); gotShape=r.shape; break; }
                default: throw new Exception("op "+op);
            }
        }
        else
        {
            NDArray r = op switch
            {
                "sum"  => np.sum(bc, (int?)ax, kd),
                "prod" => np.prod(bc, (int?)ax, (Type)null, kd),
                "min"  => np.amin(bc, (int?)ax, kd),
                "max"  => np.amax(bc, (int?)ax, kd),
                "mean" => np.mean(bc, ax, kd),
                "var"  => np.var(bc, ax, kd),
                "std"  => np.std(bc, ax, kd),
                "argmax" => np.argmax(bc, ax, kd),
                "argmin" => np.argmin(bc, ax, kd),
                "all"  => np.all(bc, ax, kd),
                "any"  => np.any(bc, ax, kd),
                "count_nonzero" => np.count_nonzero(bc, ax, kd),
                "nansum"  => np.nansum(bc, (int?)ax, kd),
                "nanmax"  => np.nanmax(bc, (int?)ax, kd),
                "nanmin"  => np.nanmin(bc, (int?)ax, kd),
                "nanmean" => np.nanmean(bc, (int?)ax, kd),
                "ptp"    => np.ptp(bc, (int?)ax, (NDArray)null, kd),
                "median" => np.median(bc, (int?)ax, (NDArray)null, false, kd),
                _ => throw new Exception("op "+op)
            };
            var rc = r.copy();
            gotShape = rc.shape;
            for (long i = 0; i < rc.size; i++) got.Add(rc.GetAtIndex(i));
        }
    }
    catch (Exception e) { err = e.GetType().Name + ":" + e.Message.Split('\n')[0]; }

    if (err != null) { fail++; fails.Add($"{id} [{dt}] THREW {err}"); continue; }

    int expN = exp.GetArrayLength();
    if (got.Count != expN) { fail++; fails.Add($"{id} [{dt}] size {got.Count}!={expN} (shape [{string.Join(",",gotShape)}] vs [{string.Join(",",expShape)}])"); continue; }

    var expArr = exp.EnumerateArray().ToArray();
    bool ok = true; string why = null;
    for (int i = 0; i < expN; i++)
    {
        object boxed = got[i];
        if (rkind == "c")
        {
            var cc = (Complex)boxed; var pe = expArr[i];
            double er = ParseF(pe[0]), ei = ParseF(pe[1]);
            double tol = 1e-6*(1+Math.Abs(er)+Math.Abs(ei));
            if (!((NaNEq(cc.Real,er)||Math.Abs(cc.Real-er)<=tol) && (NaNEq(cc.Imaginary,ei)||Math.Abs(cc.Imaginary-ei)<=tol)))
                { ok=false; why=$"[{i}] ({cc.Real},{cc.Imaginary})!=({er},{ei})"; break; }
        }
        else if (rkind == "f")
        {
            double a = ToD(boxed), b = ParseF(expArr[i]);
            double rel = dt == "half" ? 5e-2 : dt == "single" ? 1e-3 : 1e-6;  // var/std/two-pass looser
            bool eq = NaNEq(a,b) || Math.Abs(a-b) <= rel*(1+Math.Abs(b));
            if (!eq) { ok=false; why=$"[{i}] {a}!={b}"; break; }
        }
        else // i / u / b : exact
        {
            decimal a = ToDec(boxed);
            decimal b = decimal.Parse(expArr[i].GetRawText(), CultureInfo.InvariantCulture);
            if (a != b) { ok=false; why=$"[{i}] {a}!={b} (exact)"; break; }
        }
    }
    // shape: enforce for axis cases & keepdims; for flat non-keepdims accept scalar (size1)
    if (ok)
    {
        bool enforceShape = !flat || kd;
        if (enforceShape)
        {
            bool sm = gotShape.Length == expShape.Length;
            if (sm) for (int d = 0; d < gotShape.Length; d++) if (gotShape[d] != expShape[d]) { sm=false; break; }
            if (!sm) { ok=false; why=$"shape [{string.Join(",",gotShape)}]!=[{string.Join(",",expShape)}]"; }
        }
    }
    if (ok) pass++; else { fail++; fails.Add($"{id} [{dt}] {why}"); }
}

Console.WriteLine($"\n=== broadcast AXIS/arg/var/nan parity: {pass} pass, {fail} fail / {pass+fail} ===");
if (fails.Count > 0)
{
    Console.WriteLine("\n--- FAILURES grouped (op:kind) ---");
    foreach (var g in fails.GroupBy(f => {
        var head = f.Split(' ')[0]; var parts = head.Split(':');
        string opn = parts.Length>=2 ? parts[1] : head;
        string kind = f.Contains("THREW")?"THREW":f.Contains("shape")?"SHAPE":f.Contains("size")?"SIZE":"VALUE";
        return opn+":"+kind;
    }).OrderByDescending(g=>g.Count()))
        Console.WriteLine($"  {g.Key}: {g.Count()}");
    Console.WriteLine("\n--- first 100 ---");
    foreach (var f in fails.Take(100)) Console.WriteLine("  " + f);
}
