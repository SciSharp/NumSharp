#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System; using System.IO; using System.Linq; using System.Numerics; using System.Security.Cryptography; using NumSharp;
const int R=130,C=130; const string DIR=@"K:\source\NumSharp\benchmark\poc\_xref";
var refL=File.ReadAllLines(Path.Combine(DIR,"c128bool_layout.tsv")).Where(l=>l.Length>0).ToDictionary(l=>l.Split('\t')[0],l=>l.Split('\t')[1]);
NDArray Lay(NDArray b,string l)=>l switch{"C"=>b,"F"=>b.copy(order:'F'),"T"=>b.T,"sliced"=>b[$"1:{R-1}, 1:{C-1}"],"negrow"=>b["::-1, :"],"negcol"=>b[":, ::-1"],"strided"=>b[":, ::2"],"bcast"=>np.broadcast_to(b["0:1, :"],new Shape(R,C)),_=>throw new Exception(l)};
string Sha(NDArray a){var c=a.astype(NPTypeCode.Boolean,copy:true);var f=c.flat;int n=(int)c.size;var by=new byte[n];for(int i=0;i<n;i++)by[i]=(bool)f.GetAtIndex(i)?(byte)1:(byte)0;return Convert.ToHexString(SHA256.HashData(by)).ToLowerInvariant();}
var flat=new Complex[R*C];
for(int i=0;i<flat.Length;i++) flat[i]=new Complex((i%17)-8,(i%5)-2);
var sp=new (double,double)[]{(0,0),(0,3),(5,0),(double.NaN,0),(0,double.NaN),(double.PositiveInfinity,0),(-0.0,-0.0),(1e308,0)};
for(int k=0;k<sp.Length;k++) flat[(k*733+11)%flat.Length]=new Complex(sp[k].Item1,sp[k].Item2);
var baseArr=np.array(flat).reshape(R,C);
int pass=0,fail=0;
foreach(var l in new[]{"C","F","T","sliced","negrow","negcol","strided","bcast"}){
    string got=Sha(Lay(baseArr,l)); bool ok=refL.TryGetValue($"c128|{l}",out var w)&&w==got;
    if(ok)pass++; else {fail++; Console.WriteLine($"  MISMATCH c128|{l}: {got[..12]} vs {(w??"??")[..12]}");}
}
Console.WriteLine($"c128->bool layouts: {pass} pass, {fail} fail  {(fail==0?"ALL BIT-EXACT OK":"FAILED")}");
