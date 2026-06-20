#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;

const int N=1_000_000;
static double Best(Action f,int reps=20){ double b=1e9; for(int k=0;k<reps;k++){ var sw=Stopwatch.StartNew(); f(); sw.Stop(); b=Math.Min(b,sw.Elapsed.TotalMilliseconds);} return b; }

var u8=np.ones(N, NPTypeCode.Byte);
var storage=u8.Storage;
var iarr=storage.InternalArray; // IArraySlice
var block=(UnmanagedMemoryBlock<byte>)((ArraySlice<byte>)iarr).MemoryBlock;
var shp=u8.Shape;

Console.WriteLine($"N={N} (1MB), 1-byte same-type clone overhead breakdown:");
Console.WriteLine($"  full astype(same)        {Best(()=>{ var r=u8.astype(NPTypeCode.Byte); }):F4} ms");
Console.WriteLine($"  storage.Clone()          {Best(()=>{ var r=storage.Clone(); }):F4} ms");
Console.WriteLine($"  InternalArray.Clone()    {Best(()=>{ var r=iarr.Clone(); }):F4} ms");
Console.WriteLine($"  block.Clone()            {Best(()=>{ var r=block.Clone(); }):F4} ms");

unsafe {
  // raw pool.Take + MemoryCopy (the ideal memblock clone)
  Console.WriteLine($"  pool.Take(1MB) only      {Best(()=>{ var p=SizeBucketedBufferPool.Take(N); SizeBucketedBufferPool.Return(p, N); }):F4} ms");
  var dst=(byte*)SizeBucketedBufferPool.Take(N);
  Console.WriteLine($"  Buffer.MemoryCopy 1MB    {Best(()=>{ Buffer.MemoryCopy(block.Address, dst, N, N); }):F4} ms");
  // pool.Take fresh (no return -> forces new bucket alloc / first-touch)
  Console.WriteLine($"  Take+MemoryCopy+Return   {Best(()=>{ var p=(byte*)SizeBucketedBufferPool.Take(N); Buffer.MemoryCopy(block.Address,p,N,N); SizeBucketedBufferPool.Return((nint)p, N); }):F4} ms");
  Console.WriteLine($"  new Shape(_shape)        {Best(()=>{ var s=new Shape(shp); }):F4} ms");
}
