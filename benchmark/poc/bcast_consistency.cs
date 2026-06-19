#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using NumSharp;

// DECISIVE broadcast-bug detector: for every (op, dtype, layout, axis),
// compare np.op(broadcast_view) vs np.op(broadcast_view.copy()).
// The materialized copy is GROUND TRUTH for what the broadcast represents.
// Divergence => broadcast-handling bug (independent of any NumPy semantic gaps).

string[] DT = {"bool","byte","sbyte","int16","uint16","int32","uint32","int64","uint64","char","half","single","double","decimal","complex"};

static NDArray BuildBase(string dt, int[] shape, string kind)
{
    int n = 1; foreach (var s in shape) n *= s;
    double[] d = new double[n];
    for (int i = 0; i < n; i++)
    {
        if (kind == "nan")      d[i] = (new double[]{double.NaN,1.5,-2.0,double.PositiveInfinity,double.NaN,3.0,double.NegativeInfinity,0.5})[i%8];
        else if (kind == "z")   d[i] = (i%4==0) ? 0 : ((i*13+5)%40)-20;
        else                    d[i] = (((i*17+1)%23)-11)*0.5;   // plain
    }
    NDArray nd;
    switch (dt)
    {
        case "bool":   { var a=new bool[n];   for(int i=0;i<n;i++) a[i]= kind=="z" ? (i%2==0) : (i%3==0); nd=np.array(a); break; }
        case "byte":   { var a=new byte[n];   for(int i=0;i<n;i++) a[i]=(byte)(kind=="z"?(i%4==0?0:(i*7+3)%50+1):(i*7+3)%50+1); nd=np.array(a); break; }
        case "sbyte":  { var a=new sbyte[n];  for(int i=0;i<n;i++) a[i]=(sbyte)(kind=="z"&&i%4==0?0:((i*13+5)%40)-20); nd=np.array(a); break; }
        case "int16":  { var a=new short[n];  for(int i=0;i<n;i++) a[i]=(short)(kind=="z"&&i%4==0?0:((i*13+5)%40)-20); nd=np.array(a); break; }
        case "uint16": { var a=new ushort[n]; for(int i=0;i<n;i++) a[i]=(ushort)(kind=="z"&&i%4==0?0:(i*7+3)%50+1); nd=np.array(a); break; }
        case "int32":  { var a=new int[n];    for(int i=0;i<n;i++) a[i]=(kind=="z"&&i%4==0?0:((i*13+5)%40)-20); nd=np.array(a); break; }
        case "uint32": { var a=new uint[n];   for(int i=0;i<n;i++) a[i]=(uint)(kind=="z"&&i%4==0?0:(i*7+3)%50+1); nd=np.array(a); break; }
        case "int64":  { var a=new long[n];   for(int i=0;i<n;i++) a[i]=(kind=="z"&&i%4==0?0:((i*13+5)%40)-20); nd=np.array(a); break; }
        case "uint64": { var a=new ulong[n];  for(int i=0;i<n;i++) a[i]=(ulong)(kind=="z"&&i%4==0?0:(i*7+3)%50+1); nd=np.array(a); break; }
        case "char":   { var a=new char[n];   for(int i=0;i<n;i++) a[i]=(char)((i*7+3)%50+1); nd=np.array(a); break; }
        case "half":   { var a=new double[n]; for(int i=0;i<n;i++) a[i]=d[i]; nd=np.array(a).astype(NPTypeCode.Half); break; }
        case "single": { var a=new float[n];  for(int i=0;i<n;i++) a[i]=(float)d[i]; nd=np.array(a); break; }
        case "double": { var a=new double[n]; for(int i=0;i<n;i++) a[i]=d[i]; nd=np.array(a); break; }
        case "decimal":{ var a=new decimal[n];for(int i=0;i<n;i++) a[i]=(decimal)((((i*17+1)%23)-11)*0.25); nd=np.array(a); break; }
        case "complex":{ var a=new Complex[n];for(int i=0;i<n;i++) a[i]= kind=="nan" ? new Complex(i%3==0?double.NaN:i-3,1) : new Complex(((i*5)%13)-6,((i*3)%11)-5); nd=np.array(a); break; }
        default: throw new Exception(dt);
    }
    return nd.reshape(shape);
}

static NDArray Pre(NDArray v, string p) => p switch
{
    "none"  => v,
    "T"     => v.T,
    "slice" => v.ndim==2 ? v["1:-1, 1:-1"] : v["1:-1, 1:-1, 1:-1"],
    _ => throw new Exception("pre "+p)
};

static double ToD(object o) => o switch
{
    bool b => b ? 1.0 : 0.0,
    byte v => v, sbyte v => v, short v => v, ushort v => v,
    int v => v, uint v => v, long v => v, ulong v => v,
    char v => (int)v, Half v => (double)v, float v => v, double v => v, decimal v => (double)v,
    Complex z => double.NaN, _ => double.NaN
};
static bool NaNEq(double a, double b) => (double.IsNaN(a) && double.IsNaN(b)) || a == b;
static bool ValEq(object x, object y, bool floaty)
{
    if (x is Complex cx && y is Complex cy)
        return (NaNEq(cx.Real,cy.Real)||Math.Abs(cx.Real-cy.Real)<=1e-6*(1+Math.Abs(cy.Real)))
            && (NaNEq(cx.Imaginary,cy.Imaginary)||Math.Abs(cx.Imaginary-cy.Imaginary)<=1e-6*(1+Math.Abs(cy.Imaginary)));
    double a = ToD(x), b = ToD(y);
    if (!floaty) return a == b;
    return NaNEq(a,b) || Math.Abs(a-b) <= 1e-5*(1+Math.Abs(b));
}

// (tag, baseShape, target, pre)
var LAYOUTS = new (string tag,int[] bs,int[] tg,string pre)[]{
    ("a0",   new[]{1,6},    new[]{5,6},   "none"),
    ("a1",   new[]{5,1},    new[]{5,6},   "none"),
    ("ab",   new[]{1,1},    new[]{5,6},   "none"),
    ("m3",   new[]{2,1,3},  new[]{2,4,3}, "none"),
    ("o3",   new[]{1,3,4},  new[]{5,3,4}, "none"),
    ("all3", new[]{1,1,1},  new[]{3,4,2}, "none"),
    ("ncT",  new[]{4,6},    new[]{3,6,4}, "T"),
    ("ncS",  new[]{5,6},    new[]{3,3,4}, "slice"),
};

// op -> (function over an NDArray with int? axis) ; null axis = flat
Func<NDArray,int?,NDArray> R(string op) => (a,ax) => op switch
{
    "sum"  => ax.HasValue ? np.sum(a,ax,false) : np.sum(a),
    "prod" => ax.HasValue ? np.prod(a,ax,(Type)null,false) : np.prod(a),
    "min"  => ax.HasValue ? np.amin(a,ax,false) : np.amin(a),
    "max"  => ax.HasValue ? np.amax(a,ax,false) : np.amax(a),
    "mean" => ax.HasValue ? np.mean(a,ax.Value,false) : np.mean(a),
    "var"  => ax.HasValue ? np.var(a,ax.Value,false) : np.var(a),
    "std"  => ax.HasValue ? np.std(a,ax.Value,false) : np.std(a),
    "argmax" => ax.HasValue ? np.argmax(a,ax.Value,false) : NDArray.Scalar(np.argmax(a)),
    "argmin" => ax.HasValue ? np.argmin(a,ax.Value,false) : NDArray.Scalar(np.argmin(a)),
    "all"  => ax.HasValue ? (NDArray)np.all(a,ax.Value,false) : NDArray.Scalar(np.all(a)),
    "any"  => ax.HasValue ? (NDArray)np.any(a,ax.Value,false) : NDArray.Scalar(np.any(a)),
    "count_nonzero" => ax.HasValue ? np.count_nonzero(a,ax.Value,false) : NDArray.Scalar(np.count_nonzero(a)),
    "nansum"  => np.nansum(a,ax,false),
    "nanmax"  => np.nanmax(a,ax,false),
    "nanmin"  => np.nanmin(a,ax,false),
    "nanmean" => np.nanmean(a,ax,false),
    "ptp"    => np.ptp(a,ax,(NDArray)null,false),
    "median" => np.median(a,ax,(NDArray)null,false,false),
    _ => throw new Exception(op)
};

// which dtypes each op group runs over
var GROUPS = new (string[] ops, string[] dts, string kind)[]{
    (new[]{"sum","prod","min","max","mean"}, DT, "plain"),
    (new[]{"var","std"}, new[]{"byte","int32","int64","half","single","double","decimal","complex"}, "plain"),
    (new[]{"argmax","argmin"}, new[]{"bool","byte","sbyte","int16","uint16","int32","uint32","int64","uint64","half","single","double","decimal","complex"}, "plain"),
    (new[]{"all","any","count_nonzero"}, new[]{"bool","byte","int32","int64","double","complex"}, "z"),
    (new[]{"nansum","nanmax","nanmin","nanmean"}, new[]{"half","single","double","complex"}, "nan"),
    (new[]{"ptp","median"}, new[]{"byte","int32","int64","single","double"}, "plain"),
};

int pass=0, fail=0, threw=0;
var fails = new List<string>();
var threwSet = new SortedSet<string>();

foreach (var (ops, dts, kind) in GROUPS)
foreach (var op in ops)
foreach (var dt in dts)
foreach (var (tag,bs,tg,pre) in LAYOUTS)
{
    bool floaty = dt is "half" or "single" or "double" or "decimal" or "complex" || op is "mean" or "var" or "std" or "nanmean" or "median";
    NDArray baseArr;
    try { baseArr = Pre(BuildBase(dt, bs, kind), pre); } catch (Exception e) { threw++; threwSet.Add($"{op}:{dt} BUILD {e.GetType().Name}"); continue; }
    int nd = tg.Length;
    var axes = new List<int?>{ null }; for (int d=0; d<nd; d++) axes.Add(d);
    foreach (var ax in axes)
    {
        NDArray bc;
        try { bc = np.broadcast_to(baseArr, new Shape(tg)); } catch { continue; }
        NDArray viaBcast=null, viaCopy=null; string err=null;
        try { viaBcast = R(op)(bc, ax).copy(); } catch (Exception e) { err = "BCAST "+e.GetType().Name+":"+e.Message.Split('\n')[0]; }
        try { viaCopy  = R(op)(bc.copy(), ax).copy(); } catch (Exception e) { err = (err??"")+" COPY "+e.GetType().Name+":"+e.Message.Split('\n')[0]; }
        string axs = ax.HasValue ? ax.Value.ToString() : "flat";
        if (err != null)
        {
            // if BOTH throw the same way it's a dtype gap, not a broadcast bug
            if (err.Contains("BCAST") && err.Contains("COPY")) { threw++; threwSet.Add($"{op}:{dt} BOTH-THROW"); }
            else { fail++; fails.Add($"{tag}:{dt}:{op}:ax{axs} ASYMMETRIC {err}"); }
            continue;
        }
        if (viaBcast.size != viaCopy.size) { fail++; fails.Add($"{tag}:{dt}:{op}:ax{axs} SIZE {viaBcast.size}!={viaCopy.size} shapes [{string.Join(",",viaBcast.shape)}] vs [{string.Join(",",viaCopy.shape)}]"); continue; }
        bool ok=true; string why=null;
        for (long i=0;i<viaBcast.size;i++) { if (!ValEq(viaBcast.GetAtIndex(i), viaCopy.GetAtIndex(i), floaty)) { ok=false; why=$"[{i}] bcast={viaBcast.GetAtIndex(i)} copy={viaCopy.GetAtIndex(i)}"; break; } }
        if (ok) { var sb=viaBcast.shape; var sc=viaCopy.shape; if (sb.Length!=sc.Length) {ok=false;why=$"shape [{string.Join(",",sb)}]!=[{string.Join(",",sc)}]";} else for(int d=0;d<sb.Length;d++) if(sb[d]!=sc[d]){ok=false;why=$"shape [{string.Join(",",sb)}]!=[{string.Join(",",sc)}]";break;} }
        if (ok) pass++; else { fail++; fails.Add($"{tag}:{dt}:{op}:ax{axs} {why}"); }
    }
}

Console.WriteLine($"\n=== broadcast-vs-copy CONSISTENCY: {pass} pass, {fail} fail, {threw} both-throw / {pass+fail+threw} ===");
Console.WriteLine("(both-throw = dtype gap on BOTH paths => NOT a broadcast bug)");
if (threwSet.Count>0){ Console.WriteLine("\n--- symmetric dtype gaps (both throw) ---"); foreach(var t in threwSet) Console.WriteLine("  "+t); }
if (fails.Count > 0)
{
    Console.WriteLine($"\n--- BROADCAST DIVERGENCES ({fails.Count}) grouped (op:dtype) ---");
    foreach (var g in fails.GroupBy(f => { var p=f.Split(' ')[0].Split(':'); return p.Length>=3? p[1]+":"+p[2] : f; }).OrderByDescending(g=>g.Count()))
        Console.WriteLine($"  {g.Key}: {g.Count()}");
    Console.WriteLine("\n--- first 100 ---");
    foreach (var f in fails.Take(100)) Console.WriteLine("  " + f);
}
else Console.WriteLine("\nNO broadcast-specific divergences. Broadcast handling == materialized copy everywhere.");
