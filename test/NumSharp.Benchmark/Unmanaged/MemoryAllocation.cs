using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
{

    //|                      Method |         Mean |       Error |       StdDev |        Median |           Min |          Max |  Ratio | RatioSD |
    //|---------------------------- |-------------:|------------:|-------------:|--------------:|--------------:|-------------:|-------:|--------:|
    //|     UnmanagedMemoryAllocate |    380.03 us |    528.5 us |    349.56 us |    234.000 us |    225.800 us |   1,353.5 us |   1.00 |    0.00 |
    //|                      Fixing |     36.36 us |    147.2 us |     97.38 us |      5.150 us |      5.000 us |     313.5 us |   0.04 |    0.07 |
    //|                     NDArray |  1,367.49 us |  3,537.8 us |  2,340.02 us |    620.950 us |    591.000 us |   8,026.5 us |   2.79 |    1.23 |
    //|         MarshalAllocHGlobal |    164.61 us |    170.7 us |    112.90 us |    122.250 us |    121.300 us |     480.7 us |   0.49 |    0.15 |
    //|          GCHandleAllocArray |    245.72 us |    229.8 us |    152.03 us |    194.800 us |    125.400 us |     635.2 us |   0.75 |    0.31 |
    //|
    //| UnmanagedMemoryAllocate500k |    570.20 us |    576.9 us |    381.57 us |    427.050 us |    418.200 us |   1,638.5 us |   1.71 |    0.53 |
    //|                  Fixing500k |     37.31 us |    153.3 us |    101.37 us |      5.100 us |      5.000 us |     325.8 us |   0.04 |    0.07 |
    //|                 NDArray500k |  1,593.99 us |  3,567.5 us |  2,359.68 us |    836.450 us |    815.200 us |   8,308.5 us |   3.56 |    1.08 |
    //|     MarshalAllocHGlobal500k |    380.72 us |    172.4 us |    114.06 us |    338.900 us |    336.400 us |     700.7 us |   1.26 |    0.39 |
    //|      GCHandleAllocArray500k | 47,967.04 us | 11,013.0 us |  7,284.45 us | 46,759.750 us | 39,658.700 us |  60,854.3 us | 169.50 |   61.84 |
    //|
    //|   UnmanagedMemoryAllocate1m |    652.19 us |    545.7 us |    360.96 us |    514.150 us |    462.400 us |   1,660.3 us |   2.00 |    0.57 |
    //|                    Fixing1m |     44.97 us |    189.8 us |    125.52 us |      5.100 us |      5.000 us |     402.2 us |   0.05 |    0.09 |
    //|                   NDArray1m |  1,563.54 us |  3,466.7 us |  2,293.00 us |    807.800 us |    792.500 us |   8,087.0 us |   3.50 |    1.04 |
    //|       MarshalAllocHGlobal1m |    388.98 us |    222.2 us |    146.95 us |    333.550 us |    330.700 us |     801.9 us |   1.26 |    0.38 |
    //|        GCHandleAllocArray1m | 80,076.03 us | 34,242.3 us | 22,649.13 us | 77,455.100 us | 60,317.500 us | 137,219.7 us | 287.64 |  143.92 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public unsafe class MemoryAllocation
    {
        private const int iterations_small = 2_000;
        private const int iterations_large = 2_000;
        private double[] doubles100;
        private double[] doubles500_000;
        private double[] doubles1m;

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
            doubles100 = new double[100];
            doubles500_000 = new double[500_000];
            doubles1m = new double[1_000_000];
            for (int i = 0; i < 100; i++)
            {
                doubles100[i] = i;
            }

            for (int i = 0; i < 500_000; i++)
            {
                doubles500_000[i] = i;
            }

            for (int i = 0; i < 1_000_000; i++)
            {
                doubles1m[i] = i;
            }
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

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void Fixing()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                fixed (double* ptr = doubles100)
                { }
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

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void Fixing500k()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                fixed (double* ptr = doubles500_000)
                { }
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

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void Fixing1m()
        {
            for (int j = 0; j < iterations_large; j++)
            {
                fixed (double* ptr = doubles1m)
                { }
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
