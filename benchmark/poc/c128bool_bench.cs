#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System; using System.Diagnostics; using System.IO; using NumSharp;
const int R=1000,C=1000,it=30,wm=8,rd=7;
double Best(Action f){for(int i=0;i<wm;i++)f();double b=1e9;for(int r=0;r<rd;r++){var sw=Stopwatch.StartNew();for(int i=0;i<it;i++)f();b=Math.Min(b,sw.Elapsed.TotalMilliseconds/it);}return b;}
NDArray Lay(NDArray b,string l)=>l switch{"C"=>b,"F"=>b.copy(order:'F'),"T"=>b.T,"sliced"=>b[$"1:{R-1}, 1:{C-1}"],"negrow"=>b["::-1, :"],"negcol"=>b[":, ::-1"],"strided"=>b[":, ::2"],"bcast"=>np.broadcast_to(b["0:1, :"],new Shape(R,C)),_=>throw new Exception(l)};
var ba=((np.arange(R*C)%17)+1).astype(NPTypeCode.Complex).reshape(R,C);
var sb=new System.Text.StringBuilder();
foreach(var l in new[]{"C","F","T","sliced","negrow","negcol","strided","bcast"}){
    var v=Lay(ba,l); var _=v.astype(NPTypeCode.Boolean,copy:true);
    GC.Collect();GC.WaitForPendingFinalizers();GC.Collect();
    double ns=Best(()=>{var r=v.astype(NPTypeCode.Boolean,copy:true);});
    sb.AppendLine($"c128|{l}|bool\t{ns:F5}");
}
File.WriteAllText(@"K:\source\NumSharp\benchmark\poc\_xref\c128bool_ns.tsv",sb.ToString());
Console.WriteLine("done");
