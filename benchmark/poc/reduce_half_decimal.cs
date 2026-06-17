#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using NumSharp;

// Phase 3 gate: Half (vs NumPy float16 semantics via Half-sequential brute force) and
// Decimal (vs full-precision decimal brute force) axis reductions on the NpyIter path.

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.WriteLine($"[opt] core={optCore}\n");
int fails = 0;

// ---- Half: known 3x4 (%7) vs NumPy printed values ----
var hb = (np.arange(12).astype(NPTypeCode.Double).reshape(3,4) % 7).astype(NPTypeCode.Half);
CheckH("half sum axis0", np.sum(hb, axis:0), new float[]{5,8,11,7}, ref fails);
CheckH("half mean axis0", np.mean(hb, axis:0), new float[]{1.667f,2.666f,3.666f,2.334f}, ref fails);
CheckH("half prod axis0", np.prod(hb, axis:0), new float[]{0,10,36,0}, ref fails);
CheckH("half min axis1", np.amin(hb, axis:1), new float[]{0,0,1}, ref fails);
CheckH("half max axis1", np.amax(hb, axis:1), new float[]{3,6,4}, ref fails);
// 4096-ones saturation (NumPy f16 sum caps at 2048)
var ones = np.ones(new Shape(4096,2), NPTypeCode.Half);
CheckH("half sum 4096 ones (sat 2048)", np.sum(ones, axis:0), new float[]{2048,2048}, ref fails);

// ---- Half: layout matrix vs Half-sequential brute force ----
var rnd = new Random(7);
foreach (var dims in new[]{ new[]{7,5}, new[]{5,7}, new[]{4,6,3} })
{
    var baseArr = MakeHalf(dims, rnd);
    foreach (var (tag, a) in Views(baseArr, dims))
        for (int ax = 0; ax < a.ndim; ax++)
        {
            if (a.shape[ax]==1) continue;
            foreach (var op in new[]{"sum","prod","min","max","mean"})
                foreach (var kd in new[]{false,true})
                    if (!SameH(RunH(op,a,ax,kd), RefH(op,a,ax,kd), out string w))
                    { Console.WriteLine($"FAIL half {string.Join("x",dims)}/{tag} {op} ax{ax} kd{kd}: {w}"); fails++; }
        }
}

// ---- Decimal: layout matrix vs full-precision decimal brute force ----
foreach (var dims in new[]{ new[]{7,5}, new[]{4,6,3} })
{
    var baseArr = MakeDec(dims, rnd);
    foreach (var (tag, a) in Views(baseArr, dims))
        for (int ax = 0; ax < a.ndim; ax++)
        {
            if (a.shape[ax]==1) continue;
            foreach (var op in new[]{"sum","prod","min","max","mean"})
                foreach (var kd in new[]{false,true})
                    if (!SameD(RunD(op,a,ax,kd), RefD(op,a,ax,kd), out string w))
                    { Console.WriteLine($"FAIL dec {string.Join("x",dims)}/{tag} {op} ax{ax} kd{kd}: {w}"); fails++; }
        }
}

Console.WriteLine(fails==0 ? "\nALL CORRECT" : $"\n{fails} FAILURES");

// ===== helpers =====
static (string,NDArray)[] Views(NDArray b, int[] dims)
{
    var list = new System.Collections.Generic.List<(string,NDArray)>{ ("C",b), ("F",b.copy(order:'F')), ("T",b.T) };
    if (dims[0]>=4) list.Add(("S", b["::2"]));
    return list.ToArray();
}
static NDArray MakeHalf(int[] dims, Random rnd){ long n=1; foreach(var d in dims)n*=d; var fl=new float[n]; for(long i=0;i<n;i++) fl[i]=(float)Math.Round(rnd.NextDouble()*8-4,1); return np.array(fl).astype(NPTypeCode.Half).reshape(Array.ConvertAll(dims,x=>(long)x)); }
static NDArray MakeDec(int[] dims, Random rnd){ long n=1; foreach(var d in dims)n*=d; var de=new decimal[n]; for(long i=0;i<n;i++) de[i]=Math.Round((decimal)(rnd.NextDouble()*8-4),3); return np.array(de).reshape(Array.ConvertAll(dims,x=>(long)x)); }

static NDArray RunH(string op,NDArray a,int ax,bool kd)=>op switch{"sum"=>np.sum(a,axis:ax,keepdims:kd),"prod"=>np.prod(a,axis:ax,keepdims:kd),"min"=>np.amin(a,axis:ax,keepdims:kd),"max"=>np.amax(a,axis:ax,keepdims:kd),"mean"=>np.mean(a,axis:ax,keepdims:kd),_=>null};
static NDArray RunD(string op,NDArray a,int ax,bool kd)=>RunH(op,a,ax,kd);

// Half-sequential reference (matches NumPy f16 + current NumSharp).
static float[] RefH(string op,NDArray a,int axis,bool kd)
{
    int nd=a.ndim; var dims=new long[nd]; for(int i=0;i<nd;i++)dims[i]=a.shape[i]; long an=dims[axis];
    var od=new System.Collections.Generic.List<long>(); for(int i=0;i<nd;i++) if(i!=axis) od.Add(dims[i]);
    long os=1; foreach(var d in od)os*=d; var res=new float[os]; var oc=new long[od.Count];
    for(long oi=0;oi<os;oi++){ long rem=oi; for(int d=od.Count-1;d>=0;d--){oc[d]=rem%od[d];rem/=od[d];}
        var full=new long[nd]; for(int i=0,j=0;i<nd;i++) if(i!=axis) full[i]=oc[j++];
        Half acc = op switch{"sum"=>(Half)0,"mean"=>(Half)0,"prod"=>(Half)1,"min"=>Half.PositiveInfinity,"max"=>Half.NegativeInfinity,_=>(Half)0};
        double meanAcc=0; // mean accumulates in double (matches current Half mean path)
        for(long k=0;k<an;k++){ full[axis]=k; long flat=0,st=1; for(int i=nd-1;i>=0;i--){flat+=full[i]*st;st*=dims[i];} Half v=(Half)a.GetAtIndex(flat);
            switch(op){ case "sum": acc=acc+v; break; case "prod": acc=acc*v; break; case "mean": meanAcc+=(double)v; break;
                case "min": acc = Half.IsNaN(acc)?acc:(Half.IsNaN(v)?v:(v<acc?v:acc)); break;
                case "max": acc = Half.IsNaN(acc)?acc:(Half.IsNaN(v)?v:(v>acc?v:acc)); break; } }
        res[oi]= op=="mean" ? (float)(Half)(meanAcc/an) : (float)acc;
    }
    return res;
}
static decimal[] RefD(string op,NDArray a,int axis,bool kd)
{
    int nd=a.ndim; var dims=new long[nd]; for(int i=0;i<nd;i++)dims[i]=a.shape[i]; long an=dims[axis];
    var od=new System.Collections.Generic.List<long>(); for(int i=0;i<nd;i++) if(i!=axis) od.Add(dims[i]);
    long os=1; foreach(var d in od)os*=d; var res=new decimal[os]; var oc=new long[od.Count];
    for(long oi=0;oi<os;oi++){ long rem=oi; for(int d=od.Count-1;d>=0;d--){oc[d]=rem%od[d];rem/=od[d];}
        var full=new long[nd]; for(int i=0,j=0;i<nd;i++) if(i!=axis) full[i]=oc[j++];
        decimal acc = op switch{"prod"=>1m,"min"=>decimal.MaxValue,"max"=>decimal.MinValue,_=>0m};
        for(long k=0;k<an;k++){ full[axis]=k; long flat=0,st=1; for(int i=nd-1;i>=0;i--){flat+=full[i]*st;st*=dims[i];} decimal v=(decimal)a.GetAtIndex(flat);
            switch(op){ case "sum": case "mean": acc+=v; break; case "prod": acc*=v; break; case "min": if(v<acc)acc=v; break; case "max": if(v>acc)acc=v; break; } }
        res[oi]= op=="mean" ? acc/an : acc;
    }
    return res;
}
static bool SameH(NDArray got,float[] r,out string w){ w=""; if(got.size!=r.Length){w=$"size {got.size}!={r.Length}";return false;} for(long i=0;i<r.Length;i++){ float g=(float)(Half)got.GetAtIndex(i); if(!CloseF(g,r[i])){w=$"[{i}] got {g} ref {r[i]}";return false;} } return true; }
static bool SameD(NDArray got,decimal[] r,out string w){ w=""; if(got.size!=r.Length){w=$"size {got.size}!={r.Length}";return false;} for(long i=0;i<r.Length;i++){ decimal g=(decimal)got.GetAtIndex(i); if(Math.Abs(g-r[i])>0.0001m){w=$"[{i}] got {g} ref {r[i]}";return false;} } return true; }
static bool CloseF(float a,float b){ if(float.IsNaN(a)&&float.IsNaN(b))return true; if(float.IsInfinity(a)||float.IsInfinity(b))return a==b; return Math.Abs(a-b)<=0.02f*(1+Math.Abs(b)); }
static void CheckH(string name,NDArray got,float[] exp,ref int fails){ for(long i=0;i<exp.Length;i++){ float g=(float)(Half)got.GetAtIndex(i); if(!CloseF(g,exp[i])){Console.WriteLine($"FAIL {name}[{i}]: got {g} exp {exp[i]}");fails++;return;} } Console.WriteLine($"ok   {name}"); }

partial class Program {}
