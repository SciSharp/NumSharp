#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Clean best-of-7 for 1B->2B widening strided cells (SubwordWiden) incl sliced/negrow
// (ss==1 regression guard). np_ms from the sweep tsv.
//   Run: dotnet run -c Release - < benchmark/poc/subword_widen_bench.cs
using System;
using System.Diagnostics;
using NumSharp;
const int R = 1000, C = 1000, it = 25, wm = 8, rd = 7;
double Best(Action f){for(int i=0;i<wm;i++)f();double b=1e9;for(int r=0;r<rd;r++){var sw=Stopwatch.StartNew();for(int i=0;i<it;i++)f();b=Math.Min(b,sw.Elapsed.TotalMilliseconds/it);}return b;}
var TC=new System.Collections.Generic.Dictionary<string,NPTypeCode>{{"bool",NPTypeCode.Boolean},{"u8",NPTypeCode.Byte},{"i8",NPTypeCode.SByte},{"i16",NPTypeCode.Int16},{"u16",NPTypeCode.UInt16},{"char",NPTypeCode.Char}};
NDArray Lay(NDArray b,string l)=>l switch{"strided"=>b[":, ::2"],"negcol"=>b[":, ::-1"],"sliced"=>b["1:"+(b.shape[0]-1)+", 1:"+(b.shape[1]-1)],"negrow"=>b["::-1, :"],_=>throw new Exception(l)};
var NP=new System.Collections.Generic.Dictionary<string,double>{
  {"i8|strided|i16",0.115},{"u8|strided|i16",0.114},{"bool|strided|i16",0.114},{"i8|strided|char",0.115},{"u8|strided|char",0.111},
  {"i8|negcol|i16",0.24},{"u8|negcol|i16",0.24},{"bool|negcol|i16",0.24},
  {"i8|sliced|i16",0.115},{"i8|negrow|i16",0.115},  // ss==1 guards
};
Console.WriteLine($"{"cell",-20}{"ns_ms",10}{"np_ms",10}{"ratio",9}");
foreach(var kv in NP){
  var p=kv.Key.Split('|'); var s=p[0]; var l=p[1]; var d=p[2];
  var ba=(np.arange(R*C)%65521).astype(TC[s]).reshape(R,C); var v=Lay(ba,l);
  var _=v.astype(TC[d],copy:true); GC.Collect();GC.WaitForPendingFinalizers();GC.Collect();
  double ns=Best(()=>{var r=v.astype(TC[d],copy:true);});
  Console.WriteLine($"{kv.Key,-20}{ns,10:F4}{kv.Value,10:F4}{kv.Value/ns,9:F3}");
}
