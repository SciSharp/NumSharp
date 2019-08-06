using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Benchmark
{
    //|                      Method |       Mean |      Error |     StdDev |        Min |        Max |     Median |
    //|---------------------------- |-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
    //|     UnmanagedMemoryAllocate |   2.289 ms |  2.0404 ms |  1.3496 ms |   1.417 ms |   5.950 ms |   2.095 ms |
    //|         MarshalAllocHGlobal |   1.881 ms |  0.8871 ms |  0.5868 ms |   1.281 ms |   3.283 ms |   1.725 ms |
    //|          GCHandleAllocArray |   1.505 ms |  0.6911 ms |  0.4571 ms |   1.292 ms |   2.782 ms |   1.329 ms |
    //|
    //| UnmanagedMemoryAllocate500k |   3.631 ms |  1.5504 ms |  1.0255 ms |   3.288 ms |   6.549 ms |   3.297 ms |
    //|     MarshalAllocHGlobal500k |   3.188 ms |  0.1812 ms |  0.1198 ms |   3.099 ms |   3.461 ms |   3.128 ms |
    //|      GCHandleAllocArray500k | 373.787 ms | 20.6391 ms | 13.6515 ms | 367.371 ms | 412.085 ms | 368.850 ms |
    //|
    //|   UnmanagedMemoryAllocate1m |   3.863 ms |  1.4788 ms |  0.9781 ms |   3.355 ms |   6.599 ms |   3.529 ms |
    //|       MarshalAllocHGlobal1m |   3.111 ms |  0.3413 ms |  0.2258 ms |   2.943 ms |   3.675 ms |   3.029 ms |
    //|        GCHandleAllocArray1m | 584.897 ms | 13.0944 ms |  8.6612 ms | 575.367 ms | 599.496 ms | 583.491 ms |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class ArrayCloning
    {
        long[] vals = new long[15_000];

        [Benchmark]
        public void ArrayConvertClone()
        {
            var a = ArrayConvert.Clone(vals);
        }

        [Benchmark]
        public void Clone()
        {
            var a = vals.Clone();
        }
    }
}
