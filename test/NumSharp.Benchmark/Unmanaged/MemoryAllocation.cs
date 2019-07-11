using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
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
    public unsafe class MemoryAllocation
    {
        private const int iterations_small = 2_000;
        private const int iterations_large = 2_000;

        [Benchmark(Baseline = true)]
        public void UnmanagedMemoryAllocate()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var a = new UnmanagedMemoryBlock<byte>(100);
                a.Free();
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            var _ = ScalarMemoryPool.Instance;
            var __ = InfoOf<byte>.Size;
        }

        [Benchmark]
        public void NDArray()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(100)), new Shape(100));
                a.Storage.InternalArray.DangerousFree();
            }
        }

        [Benchmark]
        public void MarshalAllocHGlobal()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var a = Marshal.AllocHGlobal(100);
                Marshal.FreeHGlobal(a);
            }
        }
        
        [Benchmark]
        public void GCHandleAllocArray()
        {
            for (int j = 0; j < iterations_small; j++)
            {
                var @ref = GCHandle.Alloc(new byte[100], GCHandleType.Pinned);
                @ref.AddrOfPinnedObject();
                @ref.Free();
            }
        }
        
        [Benchmark]
        public void UnmanagedMemoryAllocate500k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new UnmanagedMemoryBlock<byte>(500_000);
                a.Free();
            }
        }
        
        [Benchmark]
        public void NDArray500k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(500_000)), new Shape(500_000));
                a.Storage.InternalArray.DangerousFree();
            }
        }
        
        [Benchmark]
        public void MarshalAllocHGlobal500k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = Marshal.AllocHGlobal(500_000);
                Marshal.FreeHGlobal(a);
            }
        }
        
        [Benchmark]
        public void GCHandleAllocArray500k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var @ref = GCHandle.Alloc(new byte[500_000], GCHandleType.Pinned);
                @ref.AddrOfPinnedObject();
                @ref.Free();
            }
        }
        
        
        [Benchmark]
        public void UnmanagedMemoryAllocate1m()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new UnmanagedMemoryBlock<byte>(1000000);
                a.Free();
            }
        }
        
        [Benchmark]
        public void NDArray1m()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(1_000_000)), new Shape(1_000_000));
                a.Storage.InternalArray.DangerousFree();
            }
        }
        
        [Benchmark]
        public void MarshalAllocHGlobal1m()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var a = Marshal.AllocHGlobal(1000000);
                Marshal.FreeHGlobal(a);
            }
        }
        
        [Benchmark]
        public void GCHandleAllocArray1m()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                var @ref = GCHandle.Alloc(new byte[1000000], GCHandleType.Pinned);
                @ref.AddrOfPinnedObject();
                @ref.Free();
            }
        }
    }
}
