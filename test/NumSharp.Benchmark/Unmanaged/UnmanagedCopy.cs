using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends.Unmanaged;
using OOMath;
using OOMath.MemoryPooling;

namespace NumSharp.Benchmark.Unmanaged
{
    //| Method | RunStrategy |     Mean |     Error |   StdDev |   Median |      Min |       Max |
    //|------- |------------ |---------:|----------:|---------:|---------:|---------:|----------:|
    //|  copy3 |   ColdStart | 82.16 ms | 1.6843 ms | 4.966 ms | 80.67 ms | 74.72 ms | 100.53 ms |
    //|  copy2 |   ColdStart | 51.93 ms | 1.6390 ms | 4.833 ms | 50.80 ms | 46.71 ms |  69.27 ms |
    //|  copy1 |   ColdStart | 82.57 ms | 1.7267 ms | 5.091 ms | 80.91 ms | 77.42 ms | 104.33 ms |
    //|   copy |   ColdStart | 81.56 ms | 1.4580 ms | 4.299 ms | 80.34 ms | 75.82 ms |  95.28 ms |

    //|  copy3 |  Throughput | 77.54 ms | 0.5398 ms | 1.441 ms | 77.34 ms | 74.97 ms |  82.12 ms |
    //|  copy2 |  Throughput | 46.63 ms | 0.7886 ms | 2.263 ms | 45.68 ms | 44.21 ms |  53.55 ms |
    //|  copy1 |  Throughput | 78.04 ms | 0.4407 ms | 1.191 ms | 77.75 ms | 75.55 ms |  82.27 ms |
    //|   copy |  Throughput | 80.73 ms | 1.1966 ms | 3.433 ms | 79.41 ms | 75.95 ms |  89.08 ms |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 100)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 100)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public unsafe class UnmanagedCopy
    {
        private const int length = 100;
        private const int iterations = 800_000;

        private UnmanagedArray<int> from;
        private UnmanagedByteStorage<int> fromvec;
        private UnmanagedArray<int> to;
        private UnmanagedByteStorage<int> setvec;

        private UnmanagedArray<int> fromsimple;
        private UnmanagedArray<int> tosimple;

        IDisposable[] memoryTrash;
        private static InternalBufferManager.PooledBufferManager pool = default;

        NDArray nd;

        [IterationSetup]
        public void Setup()
        {
            @from = new UnmanagedArray<int>(length);
            fromvec = new UnmanagedByteStorage<int>(new int[10 * length], new Shape(10, length));
            to = new UnmanagedArray<int>(length);
            setvec = new UnmanagedByteStorage<int>(Enumerable.Range(0, length).ToArray(), new Shape(length));
            nd = np.arange(length * 10).reshape(10, length);

            fromsimple = new UnmanagedArray<int>(length);
            tosimple = new UnmanagedArray<int>(length);
            pool = new InternalBufferManager.PooledBufferManager(32_777_216, 1_677_721);
        }

        [BenchmarkDotNet.Attributes.IterationCleanup()]
        public void Cleanup()
        {
            if (memoryTrash != null)
                for (var i = 0; i < memoryTrash.Length; i++)
                {
                    var vector = memoryTrash[i];
                    if (vector == null)
                        break;
                    vector.Dispose();
                    memoryTrash[i] = null;
                }

            if (pool != null)
            {
                pool.Clear();
                pool = null;
            }
        }


        [Benchmark]
        public void copy3()
        {
            for (int j = 0; j < iterations; j++)
            {
                Copy3<int>(@from).Free();
            }
        }

        [Benchmark]
        public void copy2()
        {
            for (int j = 0; j < iterations; j++)
            {
                Copy2<int>(@from).Free();
            }
        }

        [Benchmark]
        public void copy1()
        {
            for (int j = 0; j < iterations; j++)
            {
                Copy1<int>(@from).Free();
            }
        }


        [Benchmark]
        public void copy()
        {
            for (int j = 0; j < iterations; j++)
            {
                Copy<int>(@from).Free();
            }
        }

        static UnmanagedCopy() { }

        [DllImport("msvcrt.dll", EntryPoint = "memcpy", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern int MemCopy(void* dest, void* src, UIntPtr count);


        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> Copy3<T>(UnmanagedArray<T> source) where T : unmanaged
        {
            var len = source.Count * sizeof(T);
            var ret = new UnmanagedArray<T>(source.Count);
            MemCopy(ret.Address, source.Address, (UIntPtr)len);
            //Buffer.MemoryCopy(source._itemBuffer, ret._itemBuffer, len, len);
            //source.AsSpan().CopyTo(ret.AsSpan());
            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> Copy2<T>(UnmanagedArray<T> source) where T : unmanaged
        {
            var len = source.Count * sizeof(T);
            var buffer = pool.TakeBuffer(len);
            var ret = new UnmanagedArray<T>((T*)Marshal.UnsafeAddrOfPinnedArrayElement(buffer, 0), source.Count, () => pool.ReturnBuffer(buffer));
            Buffer.MemoryCopy(source.Address, ret.Address, len, len);
            //source.AsSpan().CopyTo(ret.AsSpan());
            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> Copy1<T>(UnmanagedArray<T> source) where T : unmanaged
        {
            var ret = new UnmanagedArray<T>(source.Count);
            var len = ret.Count * sizeof(T);
            Buffer.MemoryCopy(source.Address, ret.Address, len, len);
            //source.AsSpan().CopyTo(ret.AsSpan());
            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedArray<T> Copy<T>(UnmanagedArray<T> source) where T : unmanaged
        {
            var ret = new UnmanagedArray<T>(source.Count);
            source.AsSpan().CopyTo(ret.AsSpan());
            return ret;
        }
    }
}
