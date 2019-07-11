using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Benchmark.Unmanaged
{
    //|                      Method | RunStrategy |       Mean |     Error |     StdDev |     Median |        Min |        Max |
    //|---------------------------- |------------ |-----------:|----------:|-----------:|-----------:|-----------:|-----------:|
    //|     UnmanagedMemoryAllocate |   ColdStart |   1.632 ms | 0.5814 ms |  1.7142 ms |   1.443 ms |   1.372 ms |  18.591 ms |
    //|         MarshalAllocHGlobal |   ColdStart |   1.220 ms | 0.0172 ms |  0.0507 ms |   1.204 ms |   1.198 ms |   1.571 ms |
    //|          GCHandleAllocArray |   ColdStart |   1.438 ms | 0.0952 ms |  0.2807 ms |   1.326 ms |   1.250 ms |   2.960 ms |
    //
    //|     MarshalAllocHGlobal100k |   ColdStart |   3.158 ms | 0.0244 ms |  0.0720 ms |   3.132 ms |   3.118 ms |   3.507 ms |
    //|      GCHandleAllocArray100k |   ColdStart | 596.413 ms | 8.2697 ms | 24.3833 ms | 587.685 ms | 562.561 ms | 723.050 ms |
    //| UnmanagedMemoryAllocate100k |   ColdStart |   3.243 ms | 0.1094 ms |  0.3224 ms |   3.163 ms |   3.077 ms |   5.894 ms |
    //
    //|     UnmanagedMemoryAllocate |  Throughput |   1.445 ms | 0.0176 ms |  0.0497 ms |   1.433 ms |   1.372 ms |   1.600 ms |
    //|         MarshalAllocHGlobal |  Throughput |   1.304 ms | 0.0233 ms |  0.0668 ms |   1.287 ms |   1.222 ms |   1.497 ms |
    //|          GCHandleAllocArray |  Throughput |   1.309 ms | 0.0186 ms |  0.0532 ms |   1.292 ms |   1.240 ms |   1.464 ms |
    //
    //|     MarshalAllocHGlobal100k |  Throughput |   3.069 ms | 0.0318 ms |  0.0902 ms |   3.051 ms |   2.936 ms |   3.294 ms |
    //|      GCHandleAllocArray100k |  Throughput | 616.934 ms | 5.1129 ms | 14.3371 ms | 615.436 ms | 595.493 ms | 659.145 ms |
    //| UnmanagedMemoryAllocate100k |  Throughput |   3.697 ms | 0.0493 ms |  0.1407 ms |   3.680 ms |   3.504 ms |   4.081 ms |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 100)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 100)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public unsafe class MemoryAllocation
    {
        private const int iterations_small = 20_000;
        private const int iterations_large = 20_000;

        [IterationSetup]
        public void Setup() { }

        [BenchmarkDotNet.Attributes.IterationCleanup()]
        public void Cleanup() { }

        [Benchmark]
        [BenchmarkCategory("Small (100bytes)")]
        public void UnmanagedMemoryAllocate()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var a = new UnmanagedMemoryBlock<byte>(100);
                a.Free();
            }
        }

        [Benchmark]
        [BenchmarkCategory("Small (100bytes)")]
        public void MarshalAllocHGlobal()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var a = Marshal.AllocHGlobal(100);
                Marshal.FreeHGlobal(a);
            }
        }

        [Benchmark]
        [BenchmarkCategory("Small (100bytes)")]
        public void GCHandleAllocArray()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var @ref = GCHandle.Alloc(new byte[100], GCHandleType.Pinned);
                @ref.AddrOfPinnedObject();
                @ref.Free();
            }
        }

        //[Benchmark]
        //[BenchmarkCategory("Small (100bytes)")]
        //public void newbyte() {
        //    for (int j = 0; j < iterations_small; j++) {
        //        var a = new byte[100];
        //    }
        //}


        //[Benchmark]
        //[BenchmarkCategory("Small (100bytes)")]
        //public void Stackallocate() {
        //    for (int j = 0; j < iterations_small; j++) {
        //        var a = stackalloc int[100];
        //    }
        //}

        [Benchmark]
        [BenchmarkCategory("Large (100kbytes)")]
        public void MarshalAllocHGlobal100k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = Marshal.AllocHGlobal(1000000);
                Marshal.FreeHGlobal(a);
            }
        }

        [Benchmark]
        [BenchmarkCategory("Large (100kbytes)")]
        public void GCHandleAllocArray100k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var @ref = GCHandle.Alloc(new byte[1000000], GCHandleType.Pinned);
                @ref.AddrOfPinnedObject();
                @ref.Free();
            }
        }

        //[Benchmark]
        //[BenchmarkCategory("Large (100kbytes)")]
        //public void newbyte100k() {
        //    for (int j = 0; j < iterations_large; j++) {
        //        var a = new byte[100000];
        //    }
        //}


        //[Benchmark]
        //[BenchmarkCategory("Large (100kbytes)")]
        //public void Stackallocate100k() {
        //    for (int j = 0; j < iterations_large; j++) {
        //        var a = stackalloc int[100000];
        //    }
        //}

        [Benchmark]
        [BenchmarkCategory("Large (100kbytes)")]
        public void UnmanagedMemoryAllocate100k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new UnmanagedMemoryBlock<byte>(1000000);
                a.Free();
            }
        }
    }
}
